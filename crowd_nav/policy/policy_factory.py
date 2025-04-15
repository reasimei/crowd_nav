from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL, HierarchicalSARL, LLMSARL
from crowd_nav.policy.sarl import HierarchicalLLMSARL

policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['h_sarl'] = HierarchicalSARL
policy_factory['llm_sarl'] = LLMSARL
policy_factory['h_llm_sarl'] = HierarchicalLLMSARL
