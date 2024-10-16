# # main.py
# import argparse
# from train import train_model
# from test import test_model
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="CRNN OCR")
#
#     parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help="Mode: train or test")
#     parser.add_argument('--image', type=str, help="Path to the test image (required in test mode)")
#     parser.add_argument('--model', type=str, default='models/crnn_model.pth', help="Path to the saved model")
#
#     args = parser.parse_args()
#
#     if args.mode == 'train':
#         images_folder = 'C:/Users/DELL/PycharmProjects/SE173082_ChauMinhNhat/OCR/dataset/img/'
#         labels_folder = 'C:/Users/DELL/PycharmProjects/SE173082_ChauMinhNhat/OCR/dataset/annotations'
#         train_model(images_folder, labels_folder)
#     elif args.mode == 'test':
#         if not args.image:
#             raise ValueError("Image path is required in test mode")
#         test_model(args.image, args.model)