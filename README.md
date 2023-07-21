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

# Model_2 :

-> Summary of the model_2 is same as model 1 we used same architecture but we added dropout to each layer to improve performance (avoid overfitting).

## output of model 2:

EPOCH: 0
  0%|          | 0/938 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py:560: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(_create_warning_msg(
Loss=0.1526951789855957 Batch_id=937 Accuracy=89.72: 100%|██████████| 938/938 [00:43<00:00, 21.71it/s]

Test set: Average loss: 0.0675, Accuracy: 9802/10000 (98.02%)

EPOCH: 1
Loss=0.06809452921152115 Batch_id=937 Accuracy=97.33: 100%|██████████| 938/938 [00:42<00:00, 22.03it/s]

Test set: Average loss: 0.0519, Accuracy: 9831/10000 (98.31%)

EPOCH: 2
Loss=0.01710532046854496 Batch_id=937 Accuracy=97.77: 100%|██████████| 938/938 [00:43<00:00, 21.75it/s]

Test set: Average loss: 0.0411, Accuracy: 9862/10000 (98.62%)

EPOCH: 3
Loss=0.014946089126169682 Batch_id=937 Accuracy=98.14: 100%|██████████| 938/938 [00:44<00:00, 21.27it/s]

Test set: Average loss: 0.0348, Accuracy: 9887/10000 (98.87%)

EPOCH: 4
Loss=0.025842370465397835 Batch_id=937 Accuracy=98.16: 100%|██████████| 938/938 [00:43<00:00, 21.33it/s]

Test set: Average loss: 0.0320, Accuracy: 9900/10000 (99.00%)

EPOCH: 5
Loss=0.07382788509130478 Batch_id=937 Accuracy=98.37: 100%|██████████| 938/938 [00:43<00:00, 21.73it/s]

Test set: Average loss: 0.0350, Accuracy: 9884/10000 (98.84%)

EPOCH: 6
Loss=0.006195953115820885 Batch_id=937 Accuracy=98.37: 100%|██████████| 938/938 [00:43<00:00, 21.51it/s]

Test set: Average loss: 0.0268, Accuracy: 9912/10000 (99.12%)

EPOCH: 7
Loss=0.01763985864818096 Batch_id=937 Accuracy=98.43: 100%|██████████| 938/938 [00:44<00:00, 21.29it/s]

Test set: Average loss: 0.0243, Accuracy: 9924/10000 (99.24%)

EPOCH: 8
Loss=0.14123041927814484 Batch_id=937 Accuracy=98.56: 100%|██████████| 938/938 [00:42<00:00, 21.87it/s]

Test set: Average loss: 0.0243, Accuracy: 9919/10000 (99.19%)

EPOCH: 9
Loss=0.007287764456123114 Batch_id=937 Accuracy=98.50: 100%|██████████| 938/938 [00:47<00:00, 19.72it/s]

Test set: Average loss: 0.0266, Accuracy: 9913/10000 (99.13%)

EPOCH: 10
Loss=0.003910791128873825 Batch_id=937 Accuracy=98.54: 100%|██████████| 938/938 [00:49<00:00, 18.85it/s]

Test set: Average loss: 0.0251, Accuracy: 9927/10000 (99.27%)

EPOCH: 11
Loss=0.021080872043967247 Batch_id=937 Accuracy=98.69: 100%|██████████| 938/938 [00:46<00:00, 20.35it/s]

Test set: Average loss: 0.0233, Accuracy: 9932/10000 (99.32%)

EPOCH: 12
Loss=0.01667523756623268 Batch_id=937 Accuracy=98.75: 100%|██████████| 938/938 [00:43<00:00, 21.52it/s]

Test set: Average loss: 0.0230, Accuracy: 9928/10000 (99.28%)

EPOCH: 13
Loss=0.009174858219921589 Batch_id=937 Accuracy=98.73: 100%|██████████| 938/938 [00:45<00:00, 20.58it/s]

Test set: Average loss: 0.0256, Accuracy: 9926/10000 (99.26%)

EPOCH: 14
Loss=0.007584875915199518 Batch_id=937 Accuracy=98.72: 100%|██████████| 938/938 [00:43<00:00, 21.78it/s]

Test set: Average loss: 0.0212, Accuracy: 9938/10000 (99.38%)

## Output lots :

![image](https://github.com/sunandhini96/ERA_S7/assets/63030539/1a232527-c6c0-43c9-bae9-34f8ac5c0049)

## observations for model 2:

## Target:
-> We added dropout and image augmentation to improve performance and for better training. To avoid overfitting we applied dropout.
## Results:
-> Parameters: 7.7k

-> Best Train Accuracy: 98.75

-> Best Test Accuracy: 99.38

## Analysis:
-> model is training accuracy is not high as model 1 because we applied image augmentation. We are getting best test accuracy around 99.38 % but still we are not getting our target accuracy 99.4 % within 15 epochs.  








