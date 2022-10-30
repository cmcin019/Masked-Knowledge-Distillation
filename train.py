import utils
import datasets as dsets
import models.vision_transformer as vits
from models.head import iBOTHead
import parser 

args = parser.get_args_parser()

def train_MRKD(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    train_loader, _ = dsets.get_mnist_dataloaders()

    student = vits.__dict__[args.arch](
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path,  # stochastic depth
    )
    teacher = vits.__dict__[args.arch](
        patch_size=args.patch_size
    )
    embed_dim = student.embed_dim

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(
        student, 
        iBOTHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            norm_last_layer=args.norm_last_layer,
            shared_head=args.shared_head,
        )
    )

    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim, 
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()



train_MRKD(args)