from synthesis.run_constants import VLLM_GPU_MEMORY_UTILIZATION


def test_synthesis_vllm_memory_supports_the_14b_campaign_cell():
    assert VLLM_GPU_MEMORY_UTILIZATION == 0.81
