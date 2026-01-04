"""Tests for resumable dataset pattern used in checkpointable training."""

import numpy as np
import tensorflow as tf


class MockDataSource:
  """Mock data source following tfds.data_source protocol."""

  def __init__(self, size: int = 1000):
    self._size = size
    # Deterministic data: image is index repeated, label is index mod 10.
    self._data = [
        {
            'image': np.full((28, 28, 1), i % 256, dtype=np.uint8),
            'label': np.int32(i % 10),
        }
        for i in range(size)
    ]

  def __len__(self) -> int:
    return self._size

  def __getitem__(self, idx: int):
    return self._data[idx]


def create_resumable_dataset(
    data_source,
    batch_size: int,
    seed: int,
    start_index: int = 0,
):
  """Create a resumable dataset for testing."""
  dataset_len = len(data_source)
  tf_seed = tf.random.create_rng_state(seed, 'threefry')

  @tf.py_function(Tout=(tf.int64, tf.float32, tf.int32))  # pyrefly: ignore
  def get_shuffled_sample(global_idx):
    global_idx = global_idx.numpy()
    epoch = global_idx // dataset_len
    idx_in_epoch = global_idx % dataset_len
    epoch_seed = tf.random.fold_in(tf_seed, epoch)  # pyrefly: ignore
    # max_index is inclusive, so use dataset_len - 1.
    shuffled_idx = tf.random.experimental.index_shuffle(
        idx_in_epoch, epoch_seed, dataset_len - 1  # pyrefly: ignore
    )
    record = data_source[shuffled_idx]
    image = record['image'].astype(np.float32) / 255.0
    label = record['label']
    return global_idx, image, label

  ds = tf.data.Dataset.range(start_index, start_index + dataset_len * 100)
  # Use sequential map for deterministic testing (AUTOTUNE can cause issues).
  ds = ds.map(get_shuffled_sample)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(1)
  return ds, dataset_len


def test_resumable_dataset_produces_same_samples():
  """Verify that resuming from a checkpoint produces identical samples."""
  data_source = MockDataSource(size=500)
  batch_size = 32
  seed = 42

  # Run from index 0, collect first 5 batches.
  ds_full, _ = create_resumable_dataset(
      data_source, batch_size=batch_size, seed=seed, start_index=0
  )
  full_batches = []
  for global_idx, images, labels in ds_full.take(5):
    full_batches.append(
        {
            'global_idx': global_idx.numpy(),
            'images': images.numpy(),
            'labels': labels.numpy(),
        }
    )

  # Resume from after batch 2 (use the next global index).
  resume_index = int(full_batches[2]['global_idx'][-1]) + 1
  ds_resumed, _ = create_resumable_dataset(
      data_source, batch_size=batch_size, seed=seed, start_index=resume_index
  )
  resumed_batches = []
  for global_idx, images, labels in ds_resumed.take(2):
    resumed_batches.append(
        {
            'global_idx': global_idx.numpy(),
            'images': images.numpy(),
            'labels': labels.numpy(),
        }
    )

  # Verify: resumed batch 0 should match full batch 3.
  assert np.array_equal(
      full_batches[3]['global_idx'], resumed_batches[0]['global_idx']
  ), f'Global indices mismatch: {full_batches[3]["global_idx"]} vs {resumed_batches[0]["global_idx"]}'
  assert np.array_equal(
      full_batches[3]['labels'], resumed_batches[0]['labels']
  ), 'Labels mismatch for batch 3'
  assert np.allclose(
      full_batches[3]['images'], resumed_batches[0]['images']
  ), 'Images mismatch for batch 3'

  # Verify: resumed batch 1 should match full batch 4.
  assert np.array_equal(
      full_batches[4]['global_idx'], resumed_batches[1]['global_idx']
  ), 'Global indices mismatch for batch 4'
  assert np.array_equal(
      full_batches[4]['labels'], resumed_batches[1]['labels']
  ), 'Labels mismatch for batch 4'
  assert np.allclose(
      full_batches[4]['images'], resumed_batches[1]['images']
  ), 'Images mismatch for batch 4'


def test_epoch_boundaries_have_different_shuffle():
  """Verify that different epochs have different shuffle orders."""
  data_source = MockDataSource(size=100)
  dataset_len = len(data_source)
  seed = 42

  ds, _ = create_resumable_dataset(
      data_source, batch_size=dataset_len, seed=seed, start_index=0
  )

  # Get first sample of epoch 0 and epoch 1.
  batches = list(ds.take(2))
  epoch0_labels = batches[0][2].numpy()
  epoch1_labels = batches[1][2].numpy()

  # They should be different (different shuffle per epoch).
  assert not np.array_equal(
      epoch0_labels, epoch1_labels
  ), 'Epoch 0 and 1 have the same shuffle order - they should differ'


def test_same_seed_produces_same_shuffle():
  """Verify that the same seed produces identical shuffle order."""
  data_source = MockDataSource(size=500)
  batch_size = 100
  seed = 123

  ds1, _ = create_resumable_dataset(
      data_source, batch_size=batch_size, seed=seed, start_index=0
  )
  ds2, _ = create_resumable_dataset(
      data_source, batch_size=batch_size, seed=seed, start_index=0
  )

  batch1 = next(iter(ds1.take(1)))
  batch2 = next(iter(ds2.take(1)))

  assert np.array_equal(
      batch1[2].numpy(), batch2[2].numpy()  # pyrefly: ignore
  ), 'Same seed should produce identical labels'
