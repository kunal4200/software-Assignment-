1 # import resources
2. %matplotlib inline
3.
4. from PIL import Image
5. from io import BytesIO
6. import matplotlib.pyplot as plt 7. import numpy as np
8.
9. import torch
10. import torch.optim as optim
11. import requests
12. from torchvision import transforms, models
13.
14. # getting the "features" portion of VGG19
15. vgg = models.vgg19(pretrained=True).features
16.
17. # freezing all VGG parameters since we're only optimizing the target image
18. for param in vgg.parameters():
19. param.requires_grad_(False)
20.
21. ### Load in Content and Style Images
22.
23. # The `load_image` function also converts images to normalizedTensors.
24.
25. def load_image(img_path, max_size=400, shape=None):
26. ''''' Load in and transform an image, making sure the image
27. is <= 400 pixels in the x-y dims.'''
28. if "http" in img_path:
29. response = requests.get(img_path)
30. image = Image.open(BytesIO(response.content)).convert('RGB')
31. else:
32. image = Image.open(img_path).convert('RGB')
33.
34. # large images will slow down processing
35. if max(image.size) > max_size: 36. size = max_size
37. else:
38. size = max(image.size)
39.
40. if shape is not None:
41. size = shape
42.
43. in_transform = transforms.Compose([
44. transforms.Resize(size),
45. transforms.ToTensor(),
46. transforms.Normalize((0.485, 0.456, 0.406),
47. (0.229, 0.224, 0.225))])
48.
49. # discard the transparent, alpha channel (that's the :3) and add the batch
dimension\
50. image = in_transform(image)[:3,:,:].unsqueeze(0)
51.
52. return image
53.
54. # helper function for un-normalizing an image
55. # and converting it from a Tensor image to a NumPy image for
display
56. def im_convert(tensor):
57. """ Display a tensor as an image. """
58.
59. image = tensor.to("cpu").clone().detach()
60. image = image.numpy().squeeze()
61. image = image.transpose(1,2,0)
62. image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485,
0.456, 0.406))
63. image = image.clip(0, 1)
64.
65. return image
66.
67. # load in content and style image
68. content = load_image('space_needle.jpg')
69. # Resize style to match content, makes code easier
70. style = load_image('delaunay.jpg', shape=content.shape[-2:])
71.
72. # display the images
73. fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
74. # content and style ims side-by-side
75. ax1.imshow(im_convert(content)) 76. ax2.imshow(im_convert(style))
77.
78. ## Run an image forward through a model and get the features for
79. ## a set of layers. Default layers are for VGGNet matching Gatys et al (2016).
80.
81. def get_features(image, model, layers=None):
82. if layers is None:
83. layers = {'0': 'conv1_1',
84. '5': 'conv2_1',
85. '10': 'conv3_1',
86. '19': 'conv4_1',
87. '21': 'conv4_2', ## content representation 88. '28': 'conv5_1'}
89.
90. features = {}
91. x = image
92. for name, layer in model._modules.items():
93. x = layer(x)
94. if name in layers:
95. features[layers[name]] = x
96.
97. return features
98.
99. ##Calculate the Gram Matrix of a given tensor
100.
101. def gram_matrix(tensor):
102. _, d, h, w = tensor.size()
103. tensor = tensor.view(d, h * w)
104. # the style loss for one layer, weighted appropriately layer_style_loss = style_weights[layer] *
torch.mean((target_gram - style_gram)**2)
105.
106. # add to the style loss
107. style_loss += layer_style_loss / (d * h * w)108.109. #
calculate the *total* loss
110. 111. 112. 113. 114.
total_loss = content_weight * content_loss +
style_weight * style_loss
# update your target image
optimizer.zero_grad()
total_loss.backward()
115. optimizer.step() 116.
117. # display intermediate images and print the loss 118. if ii % show_every == 0:
119. print('Total loss: ', total_loss.item())
120. plt.imshow(im_convert(target))
121. plt.show()
122.
123. # display content and final, target image
124. fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10)) 125.
ax1.imshow(im_convert(content))
126. ax2.imshow(im_convert(target))
