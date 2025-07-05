import torch
import datetime
import time
from audit import forest
from audit.forest.filtering_defenses import get_defense

torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

def fun(dataset, extra_args=None):
    """
    Main function to launch poisoning jobs.

    Args:
        extra_args (dict, optional): A dictionary of additional arguments to update the default args.
                                     For example, {'dataset': 'ImageNet', 'max_epoch': 100}.
                                     Defaults to None.
    """
    # Parse input arguments
    if extra_args is not None:
        extra_args = vars(extra_args)  # 将 Namespace 转换为字典
    args = forest.options(extra_args=extra_args)



    if args.deterministic:
        forest.utils.set_deterministic()

    for iter in range(0, 1):
        setup = forest.utils.system_startup(args)

        model = forest.Victim(args, setup=setup)
        data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, dataset,
                             model.defs.mixing_method, setup=setup)
        witch = forest.Witch(args, setup=setup)

        if args.backdoor_poisoning:
            witch.patch_sources(data)

        start_time = time.time()
        if args.pretrained_model:
            print('Loading pretrained model...')
            stats_clean = None
        elif args.skip_clean_training:
            print('Skipping clean training...')
            stats_clean = None
        else:
            stats_clean = model.train(data, max_epoch=args.max_epoch)
        train_time = time.time()

        if args.poison_selection_strategy is not None:
            data.select_poisons(model, args.poison_selection_strategy)

        poison_delta = witch.brew(model, data)

        craft_time = time.time()

        filter_stats = dict()


        test_time = time.time()

        timestamps = dict(train_time=str(datetime.timedelta(seconds=train_time - start_time)).replace(',', ''),
                          craft_time=str(datetime.timedelta(seconds=craft_time - train_time)).replace(',', ''),
                          test_time=str(datetime.timedelta(seconds=test_time - craft_time)).replace(',', ''))

        # # Save run to table
        # results = (stats_clean, stats_rerun, stats_results)
        # forest.utils.record_results(data, witch.stat_optimal_loss, results,
        #                         args, model.defs, model.model_init_seed, extra_stats={**filter_stats, **timestamps})

        # Export
        data.export_poison(poison_delta, path=args.poison_path, mode=args.save)

        print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
        print('---------------------------------------------------')
        print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
        print(f'--------------------------- craft time: {str(datetime.timedelta(seconds=craft_time - train_time))}')
        print(f'--------------------------- test time: {str(datetime.timedelta(seconds=test_time - craft_time))}')
        print('-------------Job finished.-------------------------')

