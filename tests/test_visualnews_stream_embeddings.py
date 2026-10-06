"""Storage and stream contracts; live OpenCLIP/T4 validation belongs to notebook 07."""
import hashlib
import io
import json
import tarfile
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.build_visualnews_train_manifest import build_train_table
from scripts.visualnews_stream_embeddings import (
    EmbeddingCheckpoints, encode_text_embeddings, mapping_record,
    stream_image_embeddings, validate_chunk,
)


def targets(count=5):
    return pd.DataFrame({"id": range(count), "image_path": [f"visual_news/origin/bbc/images/0038/{i:03d}.jpg" for i in range(count)],
                         "caption": [f"caption {i}" for i in range(count)], "source": "bbc", "split": "train"})


def vectors(count):
    matrix = np.zeros((count,512),dtype=np.float32)
    matrix[:,0] = 1
    return matrix


def store(tmp_path, table=None, modality="image", chunk_size=2):
    if table is None:
        table = targets()
    return EmbeddingCheckpoints(tmp_path / modality, tmp_path / f"{modality}_progress.json",
                                table, "manifest-test-hash", {"encoder":"test-only"}, modality, chunk_size)


def jpeg_bytes():
    output = io.BytesIO()
    Image.new("RGB",(8,8),"red").save(output,format="JPEG")
    return output.getvalue()


def archive_fixture(tmp_path, table, corrupt_index=None):
    path = tmp_path / "images.tar"
    with tarfile.open(path,"w") as archive:
        # Unsafe unrelated member must never be extracted or matched by basename.
        bad = tarfile.TarInfo("../escape.jpg"); bad.size=1
        archive.addfile(bad,io.BytesIO(b"x"))
        for i,row in table.iterrows():
            payload = b"broken" if i==corrupt_index else jpeg_bytes()
            info = tarfile.TarInfo("./"+row.image_path.removeprefix("visual_news/")); info.size=len(payload)
            archive.addfile(info,io.BytesIO(payload))
    return path


def test_manifest_preserves_original_fields_and_rejects_mismatched_ids(tmp_path):
    table=targets(2).drop(columns="split")
    table["title"]=[None,"a title"]
    raw={str(row['id']):row for row in table.to_dict('records')}
    path=tmp_path/'train.json';path.write_text(json.dumps(raw))
    actual=build_train_table(path,expected_count=2)
    assert actual.id.tolist()==[0,1] and actual.image_path.tolist()==table.image_path.tolist()
    assert actual.caption.tolist()==table.caption.tolist() and actual.split.eq('train').all()
    assert actual.title.isna().sum()==1 and 'falsified' not in actual
    raw['0']['id']=999;path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='ID/key mismatch'):build_train_table(path,expected_count=2)


def test_single_remote_pass_checkpoint_reload_and_noop_resume(tmp_path,monkeypatch):
    table=targets();path=archive_fixture(tmp_path,table);opened=[];encoded=[]
    @contextmanager
    def remote(archive_path=None,archive_url=None):
        opened.append(archive_url)
        with tarfile.open(path,mode='r|*') as archive:yield archive,None
    monkeypatch.setattr('scripts.visualnews_stream_embeddings.open_archive',remote)
    def encode(items):encoded.append(len(items));return vectors(len(items))
    s=store(tmp_path,table)
    result=stream_image_embeddings(s,lambda image:np.asarray(image),encode,gpu_batch_size=2)
    assert opened==['https://www.cs.rice.edu/~vo9/visualnews/origin.tar']
    assert encoded==[2,2,1] and result['encoded_this_run']==5 and result['unsafe_members_skipped']==1
    assert [c['validation']['rows'] for c in s.chunks]==[2,2,1]
    for number in range(3):
        a,m,_=s.chunk_paths(number);validate_chunk(np.load(a),pd.read_parquet(m),table)
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in s.chunk_dir.iterdir()}
    reloaded=store(tmp_path,table)
    resume=stream_image_embeddings(reloaded,lambda image:None,lambda items:pytest.fail('completed GPU call'))
    assert resume['already_completed']==5 and resume['encoded_this_run']==0 and len(opened)==1
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in s.chunk_dir.iterdir()}
    assert not (tmp_path/'escape.jpg').exists()


def test_partial_restart_skips_saved_targets_and_recovers_missing_marker(tmp_path):
    table=targets();s=store(tmp_path,table)
    rows=[mapping_record(row) for row in table.iloc[:2].to_dict('records')]
    s.add(rows,vectors(2));s.chunk_paths(0)[2].unlink()
    reloaded=store(tmp_path,table)
    assert reloaded.completed_ids=={0,1} and reloaded.chunk_paths(0)[2].exists()
    path=archive_fixture(tmp_path,table);batches=[]
    result=stream_image_embeddings(reloaded,lambda image:np.asarray(image),lambda items:(batches.append(len(items)) or vectors(len(items))),
                                  archive_path=path,gpu_batch_size=2,no_match_member_limit=2)
    assert result['already_completed']==2 and result['encoded_this_run']==3 and batches==[2,1]


def test_individual_decode_failure_is_persistent_and_other_images_continue(tmp_path):
    table=targets();path=archive_fixture(tmp_path,table,corrupt_index=2);s=store(tmp_path,table)
    result=stream_image_embeddings(s,lambda image:np.asarray(image),lambda items:vectors(len(items)),archive_path=path,gpu_batch_size=2)
    assert result['failed_this_run']==1 and result['encoded_this_run']==4
    failure=json.loads(s.failures_path.read_text().strip())
    assert failure['metadata_id']==2 and failure['image_path']==table.iloc[2].image_path
    assert json.loads(s.state_path.read_text())['status']=='incomplete'


def test_model_failure_is_systemic_and_mismatch_or_corruption_rejected(tmp_path):
    table=targets();s=store(tmp_path,table);path=archive_fixture(tmp_path,table)
    def broken(items):raise RuntimeError('GPU OOM')
    with pytest.raises(RuntimeError,match='GPU OOM'):
        stream_image_embeddings(s,lambda image:np.asarray(image),broken,archive_path=path,gpu_batch_size=2)
    assert not s.failures_path.exists() and not s.completed_paths
    with pytest.raises(ValueError,match='scope mismatch'):
        EmbeddingCheckpoints(s.chunk_dir,s.state_path,table,'DIFFERENT',{'encoder':'test-only'},'image')
    s.add([mapping_record(row) for row in table.iloc[:2].to_dict('records')],vectors(2))
    a,_,_=s.chunk_paths(0);np.save(a,-vectors(2))
    with pytest.raises(ValueError,match='checksum'):store(tmp_path,table)


def test_text_is_independent_resumable_and_aligned(tmp_path):
    table=targets();s=store(tmp_path,table,modality='text');seen=[]
    def encode(captions):seen.extend(captions);return vectors(len(captions))
    result=encode_text_embeddings(s,encode,gpu_batch_size=2)
    assert seen==table.caption.tolist() and result['archive_passes_this_run']==0
    s2=store(tmp_path,table,modality='text')
    resumed=encode_text_embeddings(s2,lambda _:pytest.fail('recomputed text'))
    assert resumed['encoded_this_run']==0 and resumed['already_completed']==5


def test_invalid_norm_alignment_and_incomplete_pair_stop(tmp_path):
    table=targets();mapping=pd.DataFrame([mapping_record(row) for row in table.iloc[:2].to_dict('records')]).rename(columns={'id':'metadata_id'})
    mapping.insert(0,'embedding_row_within_chunk',[0,1])
    with pytest.raises(ValueError,match='normalized'):validate_chunk(vectors(2)*2,mapping,table)
    mapping.loc[0,'image_path']=table.iloc[1].image_path
    with pytest.raises(ValueError,match='Duplicate'):validate_chunk(vectors(2),mapping,table)
    s=store(tmp_path,table);a,_,_=s.chunk_paths(0);np.save(a,vectors(2))
    with pytest.raises(ValueError,match='incomplete checkpoint'):store(tmp_path,table)


def test_systemic_decode_failures_stop_with_persistent_log(tmp_path):
    table=targets();path=tmp_path/'broken.tar'
    with tarfile.open(path,'w') as archive:
        for row in table.to_dict('records'):
            member=tarfile.TarInfo(row['image_path'].removeprefix('visual_news/'));member.size=6
            archive.addfile(member,io.BytesIO(b'broken'))
    s=store(tmp_path,table)
    with pytest.raises(RuntimeError,match='failure safety limit'):
        stream_image_embeddings(s,lambda image:None,lambda _:pytest.fail('GPU on corrupt images'),archive_path=path)
    assert len(s.failures_path.read_text().splitlines())==5 and not s.completed_paths


def test_wrong_archive_prefix_stops_without_basename_matching(tmp_path):
    table=targets(1);path=tmp_path/'wrong.tar'
    with tarfile.open(path,'w') as archive:
        for name in ['wrong/000.jpg','wrong/001.jpg']:
            payload=jpeg_bytes();member=tarfile.TarInfo(name);member.size=len(payload)
            archive.addfile(member,io.BytesIO(payload))
    with pytest.raises(RuntimeError,match='layout unrecognized'):
        stream_image_embeddings(store(tmp_path,table),lambda image:None,lambda _:pytest.fail('wrong path GPU'),
                                archive_path=path,no_match_member_limit=2)


def production_notebook_cells():
    import nbformat
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'notebooks/07_visualnews_large_evidence_colab.ipynb'
    return [cell for cell in nbformat.read(path, as_version=4).cells if cell.cell_type == 'code']


def test_csv_failures_and_simple_progress_fields(tmp_path):
    table = targets(2)
    checkpoints = EmbeddingCheckpoints(
        tmp_path / 'image_chunks', tmp_path / 'image_embedding_progress.json',
        table, 'manifest-test-hash', {'model_name': 'ViT-B-32', 'pretrained': 'openai'},
        'image', checkpoint_size=2, failure_log_format='csv', gpu_batch_size=64,
    )
    checkpoints.record_failure(table.iloc[0].to_dict(), ValueError('broken, quoted "image"'))
    failures = pd.read_csv(checkpoints.failures_path)
    assert failures.columns.tolist() == ['id', 'image_path', 'error']
    assert failures.iloc[0].id == 0 and 'broken, quoted "image"' in failures.iloc[0].error
    progress = json.loads(checkpoints.state_path.read_text())
    expected = {'manifest_sha256', 'target_count', 'completed_count', 'chunk_count', 'failure_count',
                'archive_members_scanned', 'model', 'pretrained', 'batch_size', 'checkpoint_size'}
    assert expected.issubset(progress)
    assert progress['failure_count'] == 1 and progress['checkpoint_size'] == 2


def test_all_notebook_production_branches_disabled_by_default():
    import ast
    namespace = {'RUN_IMAGE_PRODUCTION': False, 'RUN_TEXT_PRODUCTION': False,
                 'RUN_MERGE_IMAGES': False, 'RUN_MERGE_TEXT': False, 'RUN_FINAL_VALIDATION': False}
    expected_defaults = namespace.copy()
    actual_defaults = {}
    for cell in production_notebook_cells():
        for node in ast.parse(cell.source).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in expected_defaults:
                        actual_defaults[target.id] = node.value.value
        if cell.metadata.get('production_stage'):
            # These branches need no Colab/GPU/filesystem collaborators when disabled.
            exec(compile(cell.source, 'disabled-production-cell', 'exec'), namespace)
    assert actual_defaults == expected_defaults


def notebook_namespace(tmp_path, table):
    from pathlib import Path
    import tempfile
    import warnings
    from scripts.visualnews_stream_embeddings import (
        read_validated_chunk, validate_merged_embeddings, write_json_atomic, file_sha256,
    )
    root = tmp_path / 'production'
    root.mkdir()
    namespace = {
        'np': np, 'pd': pd, 'json': json, 'hashlib': hashlib, 'Path': Path,
        'tempfile': tempfile, 'warnings': warnings, 'BytesIO': io.BytesIO,
        'Image': Image, 'UnidentifiedImageError': __import__('PIL').UnidentifiedImageError,
        'read_validated_chunk': read_validated_chunk, 'validate_merged_embeddings': validate_merged_embeddings,
        'write_json_atomic': write_json_atomic, 'file_sha256': file_sha256,
        'EmbeddingCheckpoints': EmbeddingCheckpoints, 'mapping_record': mapping_record,
        'train_manifest': table, 'manifest_sha256': 'manifest-test-hash',
        'model_config': {'model_name': 'ViT-B-32', 'pretrained': 'openai'},
        'image_chunk_dir': root / 'image_chunks', 'text_chunk_dir': root / 'text_chunks',
        'image_progress_path': root / 'image_embedding_progress.json',
        'text_progress_path': root / 'text_embedding_progress.json',
        'GPU_BATCH_SIZE': 2, 'CHECKPOINT_SIZE': 2,
        'RUN_IMAGE_PRODUCTION': False, 'RUN_TEXT_PRODUCTION': False,
        'RUN_MERGE_IMAGES': False, 'RUN_MERGE_TEXT': False, 'RUN_FINAL_VALIDATION': False,
    }
    from scripts.check_missing_assets import validate_image_path
    from scripts.extract_selected_images import canonical_member_name
    namespace.update(validate_image_path=validate_image_path, canonical_member_name=canonical_member_name)
    return namespace


def test_notebook_image_and_text_loops_one_pass_then_skip_completed(tmp_path):
    table = targets(5)
    namespace = notebook_namespace(tmp_path, table)
    path = archive_fixture(tmp_path, table)
    opened, gpu_calls = [], []

    @contextmanager
    def fixture_archive(archive_url):
        opened.append(archive_url)
        with tarfile.open(path, mode='r|*') as archive:
            yield archive, None

    namespace['open_archive'] = fixture_archive
    namespace['image_preprocess'] = lambda image: np.asarray(image)
    cells = production_notebook_cells()
    for cell in cells:
        if cell.source.startswith(('def read_image_tensor', 'def encode_image_tensors', 'def handle_image_member',
                                   'def check_member_layout', 'def finish_image_stream')):
            exec(cell.source, namespace)

    def fake_encoder(tensors):
        gpu_calls.append(len(tensors))
        return vectors(len(tensors))

    namespace['encode_image_tensors'] = fake_encoder
    namespace['RUN_IMAGE_PRODUCTION'] = True
    image_cells = [cell for cell in cells if cell.metadata.get('production_stage') == 'image']
    for cell in image_cells:
        exec(cell.source, namespace)
    assert namespace['image_run']['encoded'] == 5 and gpu_calls == [2, 2, 1]
    assert len(opened) == 1 and len(namespace['image_checkpoints'].completed_paths) == 5
    for cell in image_cells:
        exec(cell.source, namespace)
    assert len(opened) == 1 and gpu_calls == [2, 2, 1]

    seen_captions = []
    def fake_text(captions):
        seen_captions.extend(captions)
        return vectors(len(captions))
    namespace['encode_caption_batch'] = fake_text
    namespace['RUN_TEXT_PRODUCTION'] = True
    text_cells = [cell for cell in cells if cell.metadata.get('production_stage') == 'text']
    for cell in text_cells:
        exec(cell.source, namespace)
    for cell in text_cells:
        exec(cell.source, namespace)
    assert seen_captions == table.caption.tolist() and len(opened) == 1


def merge_namespace(tmp_path, table):
    namespace = notebook_namespace(tmp_path, table)
    prefixes = ('ordered_manifest =', 'def validated_chunks', 'def merge_chunks')
    for cell in production_notebook_cells():
        if cell.source.startswith(prefixes):
            exec(cell.source, namespace)
    return namespace


def add_identifiable_vectors(checkpoints, rows):
    matrix = np.zeros((len(rows), 512), dtype=np.float32)
    for number, row in enumerate(rows):
        matrix[number, row['id']] = 1.0
    checkpoints.add([mapping_record(row) for row in rows], matrix)
    checkpoints.flush()


def test_notebook_merge_common_id_order_and_final_validation(tmp_path):
    table = targets(5).iloc[[3, 1, 4, 0, 2]].reset_index(drop=True)
    namespace = merge_namespace(tmp_path, table)
    root = namespace['image_progress_path'].parent
    for modality, order in [('image', [4, 3, 1, 0, 2]), ('text', [1, 2, 3, 4, 0])]:
        s = EmbeddingCheckpoints(namespace[modality + '_chunk_dir'], namespace[modality + '_progress_path'],
                                 table, 'manifest-test-hash', namespace['model_config'], modality, 2)
        rows = table.set_index('id', drop=False).loc[order].to_dict('records')
        add_identifiable_vectors(s, rows)
        output = root / f'evidence_{modality}_embeddings.npy'
        namespace['merge_chunks'](modality, output)
        actual = np.load(output, mmap_mode='r')
        assert actual.shape == (5, 512) and actual.dtype == np.float32
        assert actual.argmax(axis=1).tolist() == [0, 1, 2, 3, 4]
        namespace[modality + '_final_path'] = output
        with pytest.raises(FileExistsError):
            namespace['merge_chunks'](modality, output)
    namespace.update(RUN_FINAL_VALIDATION=True, evidence_map_path=root / 'evidence_embedding_map.parquet',
                     large_embedding_dir=root, large_output_dir=root, frozen_pilot_hashes={})
    for cell in production_notebook_cells():
        if cell.metadata.get('production_stage') == 'validation':
            exec(cell.source, namespace)
    actual_map = pd.read_parquet(namespace['evidence_map_path'])
    assert actual_map.metadata_id.tolist() == [0, 1, 2, 3, 4]
    assert actual_map.image_path.tolist() == targets(5).image_path.tolist()
    assert json.loads((root / 'visualnews_train_final_validation.json').read_text())['mapping_rows'] == 5


def test_notebook_merge_refuses_missing_or_corrupted_targets(tmp_path):
    table = targets(3)
    namespace = merge_namespace(tmp_path, table)
    s = EmbeddingCheckpoints(namespace['image_chunk_dir'], namespace['image_progress_path'],
                             table, 'manifest-test-hash', namespace['model_config'], 'image', 2)
    add_identifiable_vectors(s, table.iloc[:2].to_dict('records'))
    output = namespace['image_progress_path'].parent / 'evidence_image_embeddings.npy'
    with pytest.raises(ValueError, match='missing production targets'):
        namespace['merge_chunks']('image', output)
    assert not output.exists()
    assert not list(output.parent.glob('tmp*.npy'))
    array, _, _ = s.chunk_paths(0)
    np.save(array, -vectors(2))
    with pytest.raises(ValueError, match='checksum mismatch'):
        namespace['merge_chunks']('image', output)
    assert not output.exists()


def test_notebook_image_failure_logs_csv_and_resumes_only_failed_image(tmp_path):
    table = targets(5)
    namespace = notebook_namespace(tmp_path, table)
    path = archive_fixture(tmp_path, table, corrupt_index=2)
    opened = []

    @contextmanager
    def fixture_archive(archive_url):
        opened.append(archive_url)
        with tarfile.open(path, mode='r|*') as archive:
            yield archive, None

    namespace.update(open_archive=fixture_archive, image_preprocess=lambda image: np.asarray(image))
    for cell in production_notebook_cells():
        if cell.source.startswith(('def read_image_tensor', 'def encode_image_tensors', 'def handle_image_member',
                                   'def check_member_layout', 'def finish_image_stream')):
            exec(cell.source, namespace)
    encoded_batch_sizes = []
    def encode(tensors):
        encoded_batch_sizes.append(len(tensors))
        return vectors(len(tensors))
    namespace.update(encode_image_tensors=encode, RUN_IMAGE_PRODUCTION=True)
    image_cells = [cell for cell in production_notebook_cells() if cell.metadata.get('production_stage') == 'image']
    with pytest.raises(RuntimeError, match='unfinished images remain'):
        for cell in image_cells:
            exec(cell.source, namespace)
    s = namespace['image_checkpoints']
    assert s.completed_ids == {0, 1, 3, 4}
    assert pd.read_csv(s.failures_path).id.tolist() == [2]
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in s.chunk_dir.iterdir()}
    archive_fixture(tmp_path, table)  # A corrected tiny fixture, not a remote rescan.
    for cell in image_cells:
        exec(cell.source, namespace)
    assert namespace['image_run']['encoded'] == 1 and len(opened) == 2
    assert encoded_batch_sizes == [2, 2, 1]
    assert namespace['image_checkpoints'].completed_ids == set(range(5))
    for filename, checksum in before.items():
        assert hashlib.sha256((s.chunk_dir / filename).read_bytes()).hexdigest() == checksum
