- threshold  /ˈθreʃhoʊld/  n. 阈，界

ROC，Receiver Operating Characteristic，受试者工作特征曲线

AUC，Area Under Curve，即 ROC 曲线下面的面积。

从AUC判断分类器（预测模型）优劣的标准：

- AUC = 1，是完美分类器，采用这个预测模型时，存在至少一个阈值能得出完美预测。绝大多数预测的场合，不存在完美分类器。
- 0.5 < AUC < 1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。
- AUC = 0.5，跟随机猜测一样（例：丢铜板），模型没有预测价值。
- AUC < 0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测。

Bonus:

既然已经这么多评价标准，为什么还要使用ROC和AUC呢？

因为ROC曲线有个很好的特性：当测试集中的正负样本的分布变化的时候，ROC曲线能够保持不变。在实际的数据集中经常会出现[类不平衡](https://zhida.zhihu.com/search?content_id=101148845&content_type=Article&match_order=1&q=类不平衡&zhida_source=entity)(class imbalance)现象，即负样本比正样本多很多(或者相反)，而且测试数据中的正负样本的分布也可能随着时间变化。

https://zhuanlan.zhihu.com/p/58587448