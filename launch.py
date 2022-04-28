""" OFA Networks.
    Example: ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
""" 
from ofa.model_zoo import ofa_net
ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
    
# Randomly sample sub-networks from OFA network
ofa_network.sample_active_subnet()
random_subnet = ofa_network.get_active_subnet(preserve_weight=True)
    
# Manually set the sub-network
ofa_network.set_active_subnet(ks=7, e=6, d=4)
manual_subnet = ofa_network.get_active_subnet(preserve_weight=True)