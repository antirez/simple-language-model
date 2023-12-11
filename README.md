Some time ago [Andrej Karpathy](https://karpathy.ai/) released this [great video](https://www.youtube.com/watch?v=kCc8FmEb1nY) where he shows step by step how to build a small language model which uses [attention](https://arxiv.org/abs/1706.03762) to predict the next character in a sequence.

I published a [video]() that shows how powerful transformers are at what they do by reimplementing the same neural network without using anything related to transofmers and in general *any prior* that we know about NLP tasks. This way we can explore both how powerful transformers are, since they provide much better results, and at the same time how incredibly versatile feed-forward networks (the foundation of it all, after all) are at handling unknown tasks.

## Permutation encoding

In the video I introduced an encoding schema that I call *permutation encoding*, as an alternative to one-hot and binary encoding (we didn't want to use embeddings becuase of the goals of the experiment). Probably I re-invented work already done, but I don't know when and if it was used previously.

Permutation encoding is an alternative to one-hot encoding (0001, 0010, 0100, 1000) and binary encoding (00, 01, 10, 11), that uses permutations of bits arrays of fixed lengths having the same amount of zeros and ones (1010, 0101, 1100, 0011). It avoids the sparsity of one-hot encoding and the biases of binary encoding at the same time. It still wastes 50% of the first hidden layer weights, but looks better than the above two alternatives.

## How to use the code in this repository

We use PyTorch. To train the network with one of the datasets contained under `datasets` just use the following command. (Note: in the command we train the NN against the Divina Commedia of Dante):

    python3 nn.py dataset/divina.txt

The training process will show the output of the network at every check point, will log the training and validation loss on a file (that you can use to plot the curves), and will also save the neural network weights in a `pth` file.

The loss log filename and weights filename will contain all the details about the number of weights, block size and so forth. You can plot the curve with something like:

    python3 plot.py loss_BA:64_BL:64_H:2048_E:12_V:77_BN:False_LR:0.0003_divina.txt

If you specify multiple training log files, the above Python program will plot the validation loss of all the training logs in the same graph. This is useful to compare different NN layouts.

Or you can generate more output (even while the network is being trained) using the saved checkpoint with:

    python3 nn.py datasets/divina.txt loss_BA:64_BL:64_H:2048_E:12_V:77_BN:False_LR:0.0003_divina.txt.pth

Note that even to just *use* the pre-trained network you need to pass the dataset, as the set of characters must be rebuilt in order to create the correct mapping.

For all the other information, please check the YouTube video.

## Example output

This is the model output trained for a few hours with the Divina Commedia.

```
>>> step 320500: train loss 1.3853, val loss 1.6271, min loss 1.5642
lle presso sente,
che l'altra piacer li dimmi e gente,
quando ver' li amar disse: «Fé di quella,
parra né radipio, per che quel forse;
ha non poco non s'avvent' ella vivostrò
secondo il suo maestro G'
Saving model  loss_BA:64_BL:64_H:2048_E:12_V:77_BN:False_LR:0.0003_divina.txt.pth
```

## License

The code is released under the MIT license.
