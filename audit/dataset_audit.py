import torch
from audit.DVBW import DVBW
from audit.DW import DW
from audit.Data_use_auditing import DUA
from audit.UBWC import UBWC
from audit.UBWP import UBWP
from audit.Zmark import Zeromark
from audit.MIA import MIA
from audit.Rapid import Rapid
from audit.DI import DI
from audit.backdoor_auditor import BackdoorAuditor


class DatasetAudit:
    """
    A class for dataset auditing, including watermark embedding and verification.
    """

    def __init__(self, args):
        pass

    def process_dataset(self, ori_dataset, aux_dataset=None):
        """
        Embed a watermark into the original dataset.

        Args:
            ori_dataset: The original dataset.
            params (dict): Additional parameters for processing.

        Returns:
         A tuple containing:
                - pub_dataset: The processed dataset with embedded watermark.
                - aux (dict): Auxiliary data required for verification.
        """
        # TODO: Implement watermark embedding logic here
        pub_dataset = ori_dataset
        aux = {}
        return pub_dataset, aux

    def verify(self, pub_dataset, model, aux: dict, aux_dataset=None):
        """
        Conduct dataset auditing to a suspicious model and output the confidence value or p-value.

        Args:
            pub_dataset: The processed dataset with embedded watermark.
            model: The model to be audited.
            aux (dict): Auxiliary data required for verification.
            params (dict): Additional parameters for verification.

        Returns:
            float: The confidence or p-value resulting from the audit.
        """
        # TODO: Implement verification logic here
        value = 0.0
        return value


    
def get_dataset_auditing(args):
    if args.audit_method == "noaudit":
        return DatasetAudit(args)
    elif args.audit_method == "DVBW":
        return DVBW(args)
    elif args.audit_method == "DW":
        return DW(args)
    elif args.audit_method == "DUA":
        return DUA(args)
    elif args.audit_method == "UBWC":
        return UBWC(args)
    elif args.audit_method == "UBWP":
        return UBWP(args)
    elif args.audit_method == "Zmark":
        return Zeromark(args)
    elif args.audit_method == "MIA":
        return MIA(args)
    elif args.audit_method == "Rapid":
        return Rapid(args)
    elif args.audit_method == "DI":
        return DI(args)
    

def auditing(args, ori_dataset, forged_dataset, model, audit_aux_dataset):
    if args.audit_method == "DVBW":
        auditor = BackdoorAuditor(args)
        return auditor.verify(ori_dataset, forged_dataset, model, args.audit_config.get("target_label"))
    elif args.audit_method == "UBW":
        auditor = BackdoorAuditor(args)
        return auditor.verify(ori_dataset, forged_dataset, model, None)
    elif args.audit_method == "Zmark":
        auditor = BackdoorAuditor(args)
        args.audit_config["clip_min"] = torch.tensor(args.audit_config["clip_min"]).to(args.device)
        args.audit_config["clip_max"] = torch.tensor(args.audit_config["clip_max"]).to(args.device)
        return auditor.verify(ori_dataset, forged_dataset, model, args.audit_config.get("target_label"), 
                              trigger=getattr(forged_dataset, "universal_perturbation", None))
    elif args.audit_method == "DW":
        auditor = BackdoorAuditor(args)
        return auditor.verify(ori_dataset, forged_dataset, model, None)
    elif args.audit_method == "DI":
        auditor = DI(args)
        return auditor.verify(forged_dataset, model, None, None)
    elif args.audit_method == "DUA":
        auditor = DUA(args)
        aux = {"published": ori_dataset,
               "unpublished": forged_dataset}
        return auditor.verify(None, model, aux, None)
    elif args.audit_method == "MIA":
        auditor = MIA(args)
        forged_dataset, aux = auditor.process_dataset(forged_dataset, audit_aux_dataset)
        return auditor.verify(forged_dataset, model, aux, audit_aux_dataset)
    elif args.audit_method == "Rapid":
        auditor = Rapid(args)
        forged_dataset, aux = auditor.process_dataset(forged_dataset, audit_aux_dataset)
        return auditor.verify(forged_dataset, model, aux, audit_aux_dataset)

