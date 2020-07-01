#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"Tree.h"
#include"Sample.h"

class RandomForest
{
public:
	RandomForest(int treeNum,int maxDepth,int minLeafSample,float minInfoGain);
	RandomForest(const char*modelPath);
	~RandomForest();
	void predict(const float*data, float&preLabel);
	void readModel(const char*path);
private:
	int _trainSampleNum;  //the total training sample number
	int _testSampleNum;  //the total testing sample number
	int _featureNum;  //the feature dimension 
	int _trainFeatureNumPerNode;  //the feature number used in a node while training
	int _treeNum;  //the number of trees
	int _maxDepth;  //the max depth which a tree can reach
	int _classNum;  //the number of classes(if regresssion,set it to 1)
	bool _isRegression;  //if it is a regression problem
	int _minLeafSample;  //terminate condition£ºthe min samples in a node
	float _minInfoGain;  //terminate condition£ºthe min information gain in a node
	Tree**_forest;//to store every tree(classification tree or regression tree)
	Sample*_trainSample;  //hold the whole trainset and some other infomation
};

#endif//RANDOM_FOREST_H
