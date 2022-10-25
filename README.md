# Masked Autoencoders Are Scalable Vision Learners

## Definitions
#### Autoencoders
- Self-supervised learning method that aims to compress and then decompress the input data
- An encoder that learns a lower dimension, latent representation of the input data (i.e. key features of an image)
- A decoder that reconstructs the encoded representation into an output that resembles the input as closely as possible
- Validation is measured by comparing the output data with the original input data
- Used in pretraining Transformers such as BERT
#### Masking
- Process of hiding part of the input data
- Different than the Mask parameter in the Transformer psuedocode
- BERT masked language modeling - you mask a word or phrase in your input, and try to predict that missing text
- For images, you can mask patches of the input image
#### Vision Transformer (ViT)
- Uses transformer architecture that is extremely similar to the NLP transformers we have studied
- The tokens are patches of an image, instead of words or subwords, but the process of token embedding and positional embedding remain the same, as well as the remaining encoder-decoder architecture
- Can be used for image classification, object detection, or even question answering about a picture

## Goal
- The authors wanted to combine those three concepts to prove that masked autoencoders are scalable self-supervised learners for computer vision, particularly vision transformers
- In summary, self-supervised pretraining occured by inputting a masked image into an autoencoder, and outputting a reconstructed image as similar to the input as possible. 

![My Image](maevl5.jpg)

- This pre-trained model can then be used for transfer learning for a variety of vision tasks down the road


## Summary
- Mask random patches of the input image and reconstruct the missing pixels. 
- Develop asymmetric encoder-decoder architecture
  - Encoder only operates on the visible, unmasked subset of patches
  - Decoder reconstructs the original image from the latent representation and mask tokens

- Masking a high percentage creates a nontrivial self supervised task
- Using this transformer architecture for this task allows you to train large models efficiently and effectively that are 3x faster and more accurate than convolutional architectures
- Therefore, it is scalable, and you can learn high capacity models that generalize very well.
- Transfer performance in downstream tasks outperforms supervised pretraining and shows promising scaling behavior.

![My Image](maevl1.jpg)


## Architecture
#### Masking
  - Divide an image into regular, non-overlapping patches
  - Randomly sample a subset of patches and mask (i.e. remove) the remaining ones
  - Using a high masking ratio eliminates redundancy, which creates a nontrivial self-supervised task that cannot be easily solved by simply extrapolating from nearby unmasked patches. This is super interesting - actually making the problem harder leads to better and faster performance. The optimal masking ratio for BERT is 15% compared to 75% here.
#### Encoder
  - Embed patches by a linear projection with added positional embeddings
  - Then process the resulting set via a series of Transformer blocks
  - However, our encoder only works on a small subset (25%) of the full image.
  - Masked patches are removed, no masked patches are required
  - This allows us to train very large encoders with only a fraction of compute and memory
#### Decoder
  - The input to the decoder is the full set of tokens consisting of encoded visible patches and mask tokens
  - Each mask token is a learned vector that indicates the precense of a missing patch to be predicted. Positional embeddings are included on the full set of patches so that the mask tokens can indicate which patches are missing.
  - Another series of transformer blocks
  - Only used during pretraining to reconstruct the image. Only the encoder is used to produce image representations
  - Decoder architecture can be flexibly designed in a manner that is independent of the encoder.
  - Decoders can be lightweight, so the full set of tokens for the reconstructed image are only processed by the decoder, which significantly reduces pre-training time. The decoder that the authors of this paper use has <10% computation per token than then encoder does
  - Each element in the decoder’s output is a vector of pixel values representing a patch
#### Reconstruction Target
  - MAE reconstructs the input by predicting the pixel values for each masked patch
  - The last layer of the decoder is a linear projection whose number of output channels equals the number of pixel values in a patch. 
  - The decoder’s output is reshaped to form a reconstructed image. 
  - Our loss function computes the mean squared error (MSE) between the reconstructed and original images in the pixel space. 
  - We compute the loss only on masked patches, similar to BERT


#### Simple Implementation
  - First, generate a token for every input patch (by linear projection and an added positional embedding)
  - Randomly shuffle list of tokens and remove the last 75% of the list
  - After encoding, append a list of mask tokens to the list of encoded patches, and unshuffle the list
  - Decoder is applied to this full list
  - No sparse operations needed
 




## Questions
- What are some other differences between masked autoencoders for images and for text?
- How could this random sampling be performed wiuthout sparse operations?


- Difference between masked autoencoding in vision and language
  - Language is information dense and requires sophisticated language understanding to predict missing words or sentences. Images have spacial redundancy and missing patches can be recovered with little high level understanding. To overcome this and encourage learning useful features, you mask a very high portion of random patches of an image. This reduces redundancy and requires a holistic understandiung
  - The decoder reconstructs pixels for images, which have lower semantic meaning, and reconstructs words for text, which have very high semantic meaning.






![My Image](maevl1.jpg)
![My Image](maevl2.jpg)
