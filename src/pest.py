import utils

# Filter future deprecation warnings
utils.filterwarnings('ignore')

root = utils.Path("../ip102_v1.1")
model_dir = utils.Path("models/pest")

detector = utils.PestDetector(root=root, model_dir=model_dir)

# Preparing data and model with an optional quantization
detector.getdata(bs=32, num_workers=1, noise=False, blur=False, basic=True)
detector.createmodel(quantize=True)

# Training model
detector.train(epochs=10, firstrun=True, min_lr=None, interpret=False)
detector.train(epochs=20, firstrun=False, min_lr=utils.asklr(), interpret=False)

# Pruning model and finetuning further
detector.loadmodel(path=model_dir / "recentbest")
detector.prunemodel(amount=0.5)
detector.findlr()
detector.train(epochs=20, firstrun=False, min_lr=utils.asklr(), interpret=True)

# Finish quantization
detector.quantize()

# Save model for mobile deployment using torchscript
detector.trace()
