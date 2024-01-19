import pytest
import itertools
import jax
import jax.numpy as jnp
from models import RNN, RNNNet


jax.config.update("jax_enable_x64", True)

@pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
def test_rnn(rnn_type: str):
    key = jax.random.PRNGKey(0)
    key, *subkey = jax.random.split(key, 2)
    input_size, hidden_size = 3, 4
    rnn_deer = RNN(input_size, hidden_size, rnn_type=rnn_type, method="deer", key=subkey[0])
    rnn_seq = RNN(input_size, hidden_size, rnn_type=rnn_type, method="sequential", key=subkey[0])
    batch_size = 2
    seq_length = 5

    input = jax.random.uniform(key, (batch_size, seq_length, input_size))
    output_deer = jax.vmap(rnn_deer)(input)
    output_seq = jax.vmap(rnn_seq)(input)

    assert output_deer.shape == output_seq.shape == (batch_size, seq_length, hidden_size)
    assert jnp.allclose(output_deer, output_seq)

@pytest.mark.parametrize("rnn_type, bidirectional, with_embedding",
                         itertools.product(["gru", "lstm"], [True, False], [True, False]))
def test_rnnnet(rnn_type: str, bidirectional: bool, with_embedding: bool):
    key = jax.random.PRNGKey(0)
    key, *subkey = jax.random.split(key, 2)
    input_size, hidden_size = 3, 4
    rnn_deer = RNNNet(input_size, hidden_size, with_embedding, rnn_type=rnn_type, method="deer",
                      bidirectional=bidirectional, key=subkey[0])
    rnn_seq = RNNNet(input_size, hidden_size, with_embedding, rnn_type=rnn_type, method="sequential",
                     bidirectional=bidirectional, key=subkey[0])
    batch_size = 2
    seq_length = 5

    if not with_embedding:
        input = jax.random.uniform(key, (batch_size, seq_length, input_size))
    else:
        input = jax.random.randint(key, (batch_size, seq_length), 0, input_size)
    output_deer = jax.vmap(rnn_deer)(input)
    output_seq = jax.vmap(rnn_seq)(input)

    assert output_deer.shape == output_seq.shape == (batch_size, seq_length, hidden_size)
    assert jnp.allclose(output_deer, output_seq)
