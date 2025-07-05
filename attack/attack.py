import logging
import copy
from attack.attack_interface import Preprocessing, Training, Postprocessing
from attack.preprocessing.filter import MedianFilterPreprocessing, GaussianFilterPreprocessing, WaveletFilterPreprocessing
from attack.preprocessing.autoencoder import AutoencoderPreprocessing
from attack.preprocessing.synthesis import ImageSynthesisPreprocessing, DPImageSynthesisPreprocessing
from attack.training.dpsgd import DPSGD
from attack.training.advtraining import AdversarialTraining
from attack.training.asd import ASD
from attack.postprocessing.outlier_detection import SVMOutlierDetection, KNNOutlierDetection
from attack.postprocessing.rs import RandomizedSmoothing
from attack.postprocessing.reprogramming import Reprogramming
from attack.postprocessing.scale_up import SCALE_UP
from attack.postprocessing.output_noise import OutputNoise, FeatureNoise


def get_attack(args):
    if args.attack_method == "noattack":
        preprocessing = Preprocessing(args)
        training = Training(args)
        postprocessing = Postprocessing(args)
    elif args.attack_method == "medianfilter":
        preprocessing = MedianFilterPreprocessing(args)
        training = Training(args)
        postprocessing = Postprocessing(args)
    elif args.attack_method == "guassianfilter":
        preprocessing = GaussianFilterPreprocessing(args)
        training = Training(args)
        postprocessing = Postprocessing(args)
    elif args.attack_method == "waveletfilter":
        preprocessing = WaveletFilterPreprocessing(args)
        training = Training(args)
        postprocessing = Postprocessing(args)
    elif args.attack_method == "autoencoder":
        preprocessing = AutoencoderPreprocessing(args)
        training = Training(args)
        postprocessing = Postprocessing(args)
    elif args.attack_method == "dpsgd":
        preprocessing = Preprocessing(args)
        training = DPSGD(args)
        postprocessing = Postprocessing(args)
    elif args.attack_method == 'advtraining':
        preprocessing = Preprocessing(args)
        training = AdversarialTraining(args)
        postprocessing = Postprocessing(args)
    elif args.attack_method == "synthesis":
        preprocessing = ImageSynthesisPreprocessing(args)
        training = Training(args)
        postprocessing = Postprocessing(args)
    elif args.attack_method == "dp_synthesis":
        preprocessing = DPImageSynthesisPreprocessing(args)
        training = Training(args)
        postprocessing = Postprocessing(args)
    elif args.attack_method == "svm_outlier_detection":
        preprocessing = Preprocessing(args)
        training = Training(args)
        postprocessing = SVMOutlierDetection(args)
    elif args.attack_method == "knn_outlier_detection":
        preprocessing = Preprocessing(args)
        training = Training(args)
        postprocessing = KNNOutlierDetection(args)
    elif args.attack_method == "randomized_smoothing":
        preprocessing = Preprocessing(args)
        training = Training(args)
        postprocessing = RandomizedSmoothing(args)
    elif args.attack_method == "reprogramming":
        preprocessing = Preprocessing(args)
        training = Training(args)
        postprocessing = Reprogramming(args)
    elif args.attack_method == "scale_up":
        preprocessing = Preprocessing(args)
        training = Training(args)
        postprocessing = SCALE_UP(args)
    elif args.attack_method == "asd":
        preprocessing = Preprocessing(args)
        training = ASD(args)
        postprocessing = Postprocessing(args)
    elif args.attack_method == "output_noise":
        preprocessing = Preprocessing(args)
        training = Training(args)
        postprocessing = OutputNoise(args)
    elif args.attack_method == "feature_noise":
        preprocessing = Preprocessing(args)
        training = Training(args)
        postprocessing = FeatureNoise(args)
    elif args.attack_method == "hybrid1":
        attack_config = copy.deepcopy(args.attack_config)
        # Get Autoencoder Config
        args.attack_config = attack_config["autoencoder"]
        preprocessing = AutoencoderPreprocessing(args)
        # Get AdvTraining Config
        args.attack_config = attack_config["advtraining"]
        training = AdversarialTraining(args)
        # Get Reprogramming Config
        args.attack_config = attack_config["reprogramming"]
        postprocessing = Reprogramming(args)
        args.attack_config = attack_config
    elif args.attack_method == "hybrid2":
        attack_config = copy.deepcopy(args.attack_config)
        args.attack_config = attack_config["waveletfilter"]
        preprocessing = WaveletFilterPreprocessing(args)
        args.attack_config = attack_config["dpsgd"]
        training = DPSGD(args)
        args.attack_config = attack_config["rs"]
        postprocessing = RandomizedSmoothing(args)
        args.attack_config = attack_config
    elif args.attack_method == "hybrid3":
        attack_config = copy.deepcopy(args.attack_config)
        args.attack_config = attack_config["waveletfilter"]
        preprocessing = WaveletFilterPreprocessing(args)
        training = Training(args)
        args.attack_config = attack_config["reprogramming"]
        postprocessing = Reprogramming(args)
        args.attack_config = attack_config
    else:
        logger = logging.getLogger(__name__)
        logger.info("No this type of attack!")
        exit(0)
    return preprocessing, training, postprocessing
