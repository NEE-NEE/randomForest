#define _CRT_SECURE_NO_WARNINGS
#include"RandomForest.h"
#include <iostream>
#define FEATURE 8

int main(int argc, const char * argv[])
{
	//static float Calc_EdgeRecognition_Score(const float vec[])
	const float vec[FEATURE] = { 244,69,2360,2560,80,1.76,7.7,1233.5 };

	float preLabel;
	RandomForest randomForest("data/minstmodel.Model");
	randomForest.predict(vec,preLabel);

	std::cout << preLabel << std::endl;
	getchar();
	return 0;
};
