from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from parameterized import parameterized
import numpy as np
import tensorflow as tf
import vtrace
import unittest

def assert_tensors_close(t1, t2):
  np.testing.assert_allclose(t1, t2.numpy(), rtol=3e-07)

def _shaped_arange(*shape):
  """Runs np.arange, converts to float and reshapes."""
  return np.arange(np.prod(shape), dtype=np.float32).reshape(*shape)


def _softmax(logits):
  """Applies softmax non-linearity on inputs."""
  return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)


def _ground_truth_calculation(discounts, log_rhos, rewards, values,
                              bootstrap_value, clip_rho_threshold,
                              clip_pg_rho_threshold):
  """Calculates the ground truth for V-trace in Python/Numpy."""
  vs = []
  seq_len = len(discounts)
  rhos = np.exp(log_rhos)
  cs = np.minimum(rhos, 1.0)
  clipped_rhos = rhos
  if clip_rho_threshold:
    clipped_rhos = np.minimum(rhos, clip_rho_threshold)
  clipped_pg_rhos = rhos
  if clip_pg_rho_threshold:
    clipped_pg_rhos = np.minimum(rhos, clip_pg_rho_threshold)

  # This is a very inefficient way to calculate the V-trace ground truth.
  # We calculate it this way because it is close to the mathematical notation of
  # V-trace.
  # v_s = V(x_s)
  #       + \sum^{T-1}_{t=s} \gamma^{t-s}
  #         * \prod_{i=s}^{t-1} c_i
  #         * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
  # Note that when we take the product over c_i, we write `s:t` as the notation
  # of the paper is inclusive of the `t-1`, but Python is exclusive.
  # Also note that np.prod([]) == 1.
  values_t_plus_1 = np.concatenate([values, bootstrap_value[None, :]], axis=0)
  for s in range(seq_len):
    v_s = np.copy(values[s])  # Very important copy.
    for t in range(s, seq_len):
      v_s += (
          np.prod(discounts[s:t], axis=0) * np.prod(cs[s:t],
                                                    axis=0) * clipped_rhos[t] *
          (rewards[t] + discounts[t] * values_t_plus_1[t + 1] - values[t]))
    vs.append(v_s)
  vs = np.stack(vs, axis=0)
  pg_advantages = (
      clipped_pg_rhos * (rewards + discounts * np.concatenate(
          [vs[1:], bootstrap_value[None, :]], axis=0) - values))

  return vtrace.VTraceReturns(vs=vs, pg_advantages=pg_advantages)


class LogProbsFromLogitsAndActionsTest(unittest.TestCase):

  @parameterized.expand([('Batch1',1),('Batch2',2)])
  def test_log_probs_from_logits_and_actions(self, _, batch_size):
    """Tests log_probs_from_logits_and_actions."""
    seq_len = 7
    num_actions = 3

    policy_logits = _shaped_arange(seq_len, batch_size, num_actions) + 10
    actions = np.random.randint(
        0, num_actions, size=(seq_len, batch_size), dtype=np.int32)

    action_log_probs_tensor = vtrace.log_probs_from_logits_and_actions(
        policy_logits, actions)

    # Ground Truth
    # Using broadcasting to create a mask that indexes action logits
    action_index_mask = actions[..., None] == np.arange(num_actions)

    def index_with_mask(array, mask):
      return array[mask].reshape(*array.shape[:-1])

    # Note: Normally log(softmax) is not a good idea because it's not
    # numerically stable. However, in this test we have well-behaved values.
    ground_truth_v = index_with_mask(
        np.log(_softmax(policy_logits)), action_index_mask)

    assert_tensors_close(ground_truth_v, action_log_probs_tensor)


class VtraceTest(unittest.TestCase):

  @parameterized.expand([('Batch1',1), ('Batch2',5)])
  def test_vtrace(self, _, batch_size):
    """Tests V-trace against ground truth data calculated in python."""
    seq_len = 5

    # Create log_rhos such that rho will span from near-zero to above the
    # clipping thresholds. In particular, calculate log_rhos in [-2.5, 2.5),
    # so that rho is in approx [0.08, 12.2).
    log_rhos = _shaped_arange(seq_len, batch_size) / (batch_size * seq_len)
    log_rhos = 5 * (log_rhos - 0.5)  # [0.0, 1.0) -> [-2.5, 2.5).
    values = {
        'log_rhos': log_rhos,
        # T, B where B_i: [0.9 / (i+1)] * T
        'discounts':
            np.array([[0.9 / (b + 1)
                       for b in range(batch_size)]
                      for _ in range(seq_len)]),
        'rewards':
            _shaped_arange(seq_len, batch_size),
        'values':
            _shaped_arange(seq_len, batch_size) / batch_size,
        'bootstrap_value':
            _shaped_arange(batch_size) + 1.0,
        'clip_rho_threshold':
            3.7,
        'clip_pg_rho_threshold':
            2.2,
    }

    output = vtrace.from_importance_weights(**values)

    output_v = output

    ground_truth_v = _ground_truth_calculation(**values)
    for a, b in zip(ground_truth_v, output_v):
      assert_tensors_close(a, b)

  @parameterized.expand([('Batch1',1), ('Batch2',2)])
  def test_vtrace_from_logits(self,_,batch_size):
    """Tests V-trace calculated from logits."""
    seq_len = 5
    num_actions = 3
    clip_rho_threshold = None  # No clipping.
    clip_pg_rho_threshold = None  # No clipping.

    # Intentionally leaving shapes unspecified to test if V-trace can
    # deal with that.
    placeholders = {
        'behaviour_policy_logits':
            _shaped_arange(seq_len, batch_size, num_actions),
        'target_policy_logits':
            _shaped_arange(seq_len, batch_size, num_actions),
        'actions':
            np.random.randint(0, num_actions - 1, size=(seq_len, batch_size)),
        'discounts':
            np.array(  # T, B where B_i: [0.9 / (i+1)] * T
                [[0.9 / (b + 1)
                  for b in range(batch_size)]
                 for _ in range(seq_len)]),
        'rewards':
            _shaped_arange(seq_len, batch_size),
        'values':
            _shaped_arange(seq_len, batch_size) / batch_size,
        'bootstrap_value':
            _shaped_arange(batch_size) + 1.0,  # B
    }

    from_logits_output = vtrace.from_logits(
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
        **placeholders)

    target_log_probs = vtrace.log_probs_from_logits_and_actions(
        placeholders['target_policy_logits'], placeholders['actions'])
    behaviour_log_probs = vtrace.log_probs_from_logits_and_actions(
        placeholders['behaviour_policy_logits'], placeholders['actions'])
    log_rhos = target_log_probs - behaviour_log_probs
    ground_truth = (log_rhos, behaviour_log_probs, target_log_probs)


    #feed_dict = {placeholders[k]: v for k, v in placeholders.items()}

    from_logits_output_v = vtrace.from_logits(
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
        **placeholders)

    (ground_truth_log_rhos, ground_truth_behaviour_action_log_probs,
    ground_truth_target_action_log_probs) = ground_truth

    # Calculate V-trace using the ground truth logits.
    from_iw = vtrace.from_importance_weights(
        log_rhos=ground_truth_log_rhos,
        discounts=placeholders['discounts'],
        rewards=placeholders['rewards'],
        values=placeholders['values'],
        bootstrap_value=placeholders['bootstrap_value'],
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold)


    from_iw_v = from_iw

    assert_tensors_close(from_iw_v.vs, from_logits_output_v.vs)
    assert_tensors_close(from_iw_v.pg_advantages,
                        from_logits_output_v.pg_advantages)
    assert_tensors_close(ground_truth_behaviour_action_log_probs,
                        from_logits_output_v.behaviour_action_log_probs)
    assert_tensors_close(ground_truth_target_action_log_probs,
                        from_logits_output_v.target_action_log_probs)
    assert_tensors_close(ground_truth_log_rhos, from_logits_output_v.log_rhos)

  @parameterized.expand([('Batch1',1,3), ('Batch2',2,4)])
  def test_higher_rank_inputs_for_importance_weights(self,_,B,T):
    """Checks support for additional dimensions in inputs."""
    placeholders = {
        'log_rhos': _shaped_arange(T,B, 1),
        'discounts': _shaped_arange(T,B, 1),
        'rewards': _shaped_arange(T,B, 42),
        'values': _shaped_arange(T,B, 42),
        'bootstrap_value': _shaped_arange(B, 42)
    }
    output = vtrace.from_importance_weights(**placeholders)
    self.assertEqual(output.vs.shape.as_list()[-1], 42)

  @parameterized.expand([('Batch1',1,3), ('Batch2',2,4)])
  def test_inconsistent_rank_inputs_for_importance_weights(self,_,B,T):
    """Test one of many possible errors in shape of inputs."""
    placeholders = {
        'log_rhos': _shaped_arange(T,B, 1),
        'discounts': _shaped_arange(T,B, 1),
        'rewards': _shaped_arange(T, B, 42),
        'values': _shaped_arange(T, B, 42),
        'bootstrap_value': _shaped_arange(B)
    }
    with self.assertRaisesRegexp(ValueError, 'must have rank 2'):
      vtrace.from_importance_weights(**placeholders)


if __name__ == '__main__':
  unittest.main(failfast=True)