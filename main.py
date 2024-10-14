
import sys
import logging
import config
# from dataset_tools.loader import DatasetLoader
# from dataset_tools.labeler import DatasetLabeler
# from unittests.dataset_generator import generate_temporary_dataset

if __name__ == "__main__":
  # root_path, subsets, cleanup_callback = generate_temporary_dataset(sys.argv[1])

  # loader = DatasetLoader(root_path, load_labels=False, load_label_data=False)
  # labeler = DatasetLabeler(loader)
  # labeler.process_images()
  # cleanup_callback()

  config.logger.configure_logging()
  parser = config.parser.configure_parser()
  args = parser.parse_args()

  print(args)

  if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)

  if args.label and args.train:
    logging.error("You cannot label and train at the same time.")
    sys.exit(1)

  if not args.label and not args.train:
    logging.error("You must specify either --label or --train.")
    sys.exit(1)

  if not args.path and not args.dataset and not args.model:
    logging.error("A Dataset|Model path must be specified using --dataset, --model or as a positional argument.")

  if args.label:
    logging.info("Labeling images...")
    # labeler = DatasetLabeler(loader)
    # labeler.process_images()
    logging.info("Done.")
