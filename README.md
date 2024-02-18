# RoBELTa

The BELT approach

The RoBERTa model can process texts of the maximal length of 512 tokens (roughly speaking tokens are equivalent to words). It is a consequence of the model architecture and cannot be directly adjusted. Method to overcome this issue was proposed by Devlin (one of the authors of BERT). The main goal of this project is to implement this method and allow the RoBERTa model to process longer texts during prediction and fine-tuning. We dub this approach BELT (BERT For Longer Texts).

This uses any pre-trained RoBERTa model, with sending the input as batch containing mini batches to process the whole input.
