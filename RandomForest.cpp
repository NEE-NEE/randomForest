#define _CRT_SECURE_NO_WARNINGS
#include"RandomForest.h"

RandomForest::RandomForest(int treeNum,int maxDepth,int minLeafSample,float minInfoGain)
{
	_treeNum=treeNum;
	_maxDepth=maxDepth;
	_minLeafSample=minLeafSample;
	_minInfoGain=minInfoGain;
	_trainSample=NULL;
 
	_forest=new Tree*[_treeNum];
	for(int i=0;i<_treeNum;++i)
	{_forest[i]=NULL;}
}

RandomForest::RandomForest(const char*modelPath)
{
	readModel(modelPath);
}

RandomForest::~RandomForest()
{
	if(_forest!=NULL)
	{
		for(int i=0;i<_treeNum;++i)
		{
			if(_forest[i]!=NULL)
			{
				delete _forest[i];
				_forest[i]=NULL;
			}
		}
		delete[] _forest;
		_forest=NULL;
	}
	if(_trainSample!=NULL)
	{
		delete _trainSample;
		_trainSample=NULL;
	}
}


void RandomForest::predict(const float*data,float&preLabel)
{
	//get the predict from every tree
	//if regression,_classNum=1
	float*result=new float[_classNum];
	int i=0;
	for(i=0;i<_classNum;++i)
	{result[i]=0;}
	for(i=0;i<_treeNum;++i)//_treeNum
	{
		Result r;
		r.label=0;
		r.prob=0;//Result 
		r=_forest[i]->predict(data);
		result[static_cast<int>(r.label)]+=r.prob;
	}
	if(_isRegression)
	{
		preLabel=result[0]/_treeNum;}
	else
	{
		float maxProbLabel=0;
		float maxProb=result[0];
		for(i=1;i<_classNum;++i)
		{
			if(result[i]>maxProb)
			{
				maxProbLabel=i;
				maxProb=result[i];
			}
		}
		preLabel = maxProbLabel;
		//preProb = maxProb;
	}
	delete[] result;
}

void RandomForest::readModel(const char*path)
{
	_minLeafSample=0;
	_minInfoGain=0;
	_trainFeatureNumPerNode=0;
	FILE* modelFile=fopen(path,"rb");
	fread(&_treeNum,sizeof(int),1,modelFile);
	fread(&_maxDepth,sizeof(int),1,modelFile);
	fread(&_classNum,sizeof(int),1,modelFile);
	fread(&_isRegression,sizeof(bool),1,modelFile);
	int nodeNum=static_cast<int>(pow(2.0,_maxDepth)-1);
	_trainSample=NULL;
	_forest=new Tree*[_treeNum];
	//initialize every tree
	for(int i=0;i<_treeNum;++i)
	{
		_forest[i]=new ClasTree(_maxDepth,_trainFeatureNumPerNode,
			_minLeafSample,_minInfoGain,_isRegression);
	}

	int*nodeTable=new int[nodeNum];
	int isLeaf=-1;
	int featureIndex=0;
	float threshold=0;
	float value=0;
	float clas=0;
	float prob=0;
	for(int i=0;i<_treeNum;++i)
	{
		memset(nodeTable,0,sizeof(int)*nodeNum);
		nodeTable[0]=1;
		for(int j=0;j<nodeNum;j++)
		{
			//if current node is marked as null,continue
			if(nodeTable[j]==0)
			{continue;}
			fread(&isLeaf,sizeof(int),1,modelFile);
			if(isLeaf==0)  //split node
			{
				nodeTable[j*2+1]=1;
				nodeTable[j*2+2]=1;
				fread(&featureIndex,sizeof(int),1,modelFile);
				fread(&threshold,sizeof(float),1,modelFile);
				_forest[i]->createNode(j,featureIndex,threshold);
			}
			else if(isLeaf==1)  //leaf
			{
				if(_isRegression)
				{
					fread(&value,sizeof(float),1,modelFile);
					((RegrTree*)_forest[i])->createLeaf(j,value);
				}
				else
				{
					fread(&clas,sizeof(float),1,modelFile);
					fread(&prob,sizeof(float),1,modelFile);
					((ClasTree*)_forest[i])->createLeaf(j,clas,prob);
				}
			}
		}
		//fread(&isLeaf,sizeof(int),1,modelFile);
	}
	fclose(modelFile);
	delete[] nodeTable;
}
