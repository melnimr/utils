from unet_models import Loss, UNet11

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--dice-weight', type=float)
    arg('--nll-weights', action='store_true')
    arg('--device-ids', type=str, help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--size', type=str, default='1280x1920', help='Input size, for example 288x384. Must be multiples of 32')
    utils.add_args(parser)
    args = parser.parse_args()

    model_name = 'unet_11'

    args.root = str(utils.MODEL_PATH / model_name)

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    model = UNet11()

    device_ids = list(map(int, args.device_ids.split(',')))
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    loss = Loss()
    
    
    if __name__ == '__main__':
    main()
