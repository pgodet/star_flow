from . import pwcnet
from . import pwcnet_irr

from . import pwcnet_occ_joint
from . import pwcnet_irr_occ_joint

from . import tr_flow
from . import tr_features

from . import IRR_PWC
from . import IRR_PWC_occ_joint
from . import STAR

PWCNet               = pwcnet.PWCNet
PWCNet_irr           = pwcnet_irr.PWCNet
PWCNet_occ_joint           = pwcnet_occ_joint.PWCNet
PWCNet_irr_occ_joint = pwcnet_irr_occ_joint.PWCNet

TRFlow = tr_flow.TRFlow
TRFlow_occjoint = tr_flow.TRFlow_occjoint
TRFlow_irr = tr_flow.TRFlow_irr
TRFlow_irr_occjoint = tr_flow.TRFlow_irr_occjoint

TRFeat = tr_features.TRFeat
TRFeat_occjoint = tr_features.TRFeat_occjoint
TRFeat_irr_occjoint = tr_features.TRFeat_irr_occjoint

# -- With refinement ---

IRR_PWC = IRR_PWC.PWCNet
IRR_occ_joint = IRR_PWC_occ_joint.PWCNet

StarFlow = STAR.StarFlow
