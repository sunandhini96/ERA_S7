# ERA_S7

## Target: 15 epochs, test accuracy 99.4 %, and number of parameters 8K

model.py file contains Model_1, Model_2 and Model_3 classes


# Model_1 :

## Summary of the model_1 :

<img width="467" alt="image" src="https://github.com/sunandhini96/ERA_S7/assets/63030539/54b189ac-2d34-458a-8468-9b9dc6d965d6">

## Output of the model_1:

EPOCH: 0
Loss=0.04942340403795242 Batch_id=937 Accuracy=92.77: 100%|██████████| 938/938 [00:20<00:00, 45.38it/s]

Test set: Average loss: 0.1007, Accuracy: 9708/10000 (97.08%)

EPOCH: 1
Loss=0.05280555412173271 Batch_id=937 Accuracy=98.18: 100%|██████████| 938/938 [00:19<00:00, 47.22it/s]

Test set: Average loss: 0.0368, Accuracy: 9887/10000 (98.87%)

EPOCH: 2
Loss=0.008415513671934605 Batch_id=937 Accuracy=98.59: 100%|██████████| 938/938 [00:20<00:00, 44.82it/s]

Test set: Average loss: 0.0337, Accuracy: 9904/10000 (99.04%)

EPOCH: 3
Loss=0.03257976472377777 Batch_id=937 Accuracy=98.81: 100%|██████████| 938/938 [00:19<00:00, 47.17it/s]

Test set: Average loss: 0.0314, Accuracy: 9906/10000 (99.06%)

EPOCH: 4
Loss=0.05680623650550842 Batch_id=937 Accuracy=98.85: 100%|██████████| 938/938 [00:20<00:00, 46.19it/s]

Test set: Average loss: 0.0360, Accuracy: 9887/10000 (98.87%)

EPOCH: 5
Loss=0.009116838686168194 Batch_id=937 Accuracy=98.96: 100%|██████████| 938/938 [00:21<00:00, 44.23it/s]

Test set: Average loss: 0.0283, Accuracy: 9920/10000 (99.20%)

EPOCH: 6
Loss=0.040418628603219986 Batch_id=937 Accuracy=99.12: 100%|██████████| 938/938 [00:21<00:00, 43.16it/s]

Test set: Average loss: 0.0239, Accuracy: 9921/10000 (99.21%)

EPOCH: 7
Loss=0.012258580885827541 Batch_id=937 Accuracy=99.15: 100%|██████████| 938/938 [00:20<00:00, 44.82it/s]

Test set: Average loss: 0.0255, Accuracy: 9915/10000 (99.15%)

EPOCH: 8
Loss=0.005347209516912699 Batch_id=937 Accuracy=99.17: 100%|██████████| 938/938 [00:19<00:00, 47.32it/s]

Test set: Average loss: 0.0248, Accuracy: 9923/10000 (99.23%)

EPOCH: 9
Loss=0.002950513269752264 Batch_id=937 Accuracy=99.20: 100%|██████████| 938/938 [00:20<00:00, 46.30it/s]

Test set: Average loss: 0.0334, Accuracy: 9889/10000 (98.89%)

EPOCH: 10
Loss=0.03483585640788078 Batch_id=937 Accuracy=99.29: 100%|██████████| 938/938 [00:21<00:00, 44.11it/s]

Test set: Average loss: 0.0241, Accuracy: 9930/10000 (99.30%)

EPOCH: 11
Loss=0.03235429897904396 Batch_id=937 Accuracy=99.30: 100%|██████████| 938/938 [00:21<00:00, 44.31it/s]

Test set: Average loss: 0.0288, Accuracy: 9910/10000 (99.10%)

EPOCH: 12
Loss=0.0029859875794500113 Batch_id=937 Accuracy=99.31: 100%|██████████| 938/938 [00:19<00:00, 47.40it/s]

Test set: Average loss: 0.0204, Accuracy: 9934/10000 (99.34%)

EPOCH: 13
Loss=0.006023637484759092 Batch_id=937 Accuracy=99.36: 100%|██████████| 938/938 [00:20<00:00, 44.75it/s]

Test set: Average loss: 0.0220, Accuracy: 9932/10000 (99.32%)

EPOCH: 14
Loss=0.023460283875465393 Batch_id=937 Accuracy=99.43: 100%|██████████| 938/938 [00:21<00:00, 44.12it/s]

Test set: Average loss: 0.0219, Accuracy: 9932/10000 (99.32%)

## Output plots :

![image](https://github.com/sunandhini96/ERA_S7/assets/63030539/900f708a-88b2-417e-b876-56024cd289ce)

## Observations for model_1 : 

## Target:
-> Created the model skeleton and also applied batch normalization 
## Results:
-> Parameters: 7.7k

-> Best Train Accuracy: 99.43

-> Best Test Accuracy: 99.34 (12th Epoch)

## Analysis:
-> model is underfitting at starting but after few epochs model is overfitting. We are getting best test accuracy around 99.34 % but still we are not getting our target accuracy 99.4 % within 15 epochs.  







