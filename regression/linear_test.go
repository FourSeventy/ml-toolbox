package regression

import (
	_ "fmt"
	"github.com/fourseventy/ml-toolbox/dataset"
	"testing"
)

var hypTests = []struct {
	theta0 float64
	theta1 float64
	x      float64
	y      float64
}{
	//both thetas zero
	{0, 0, 0, 0},
	{0, 0, 4, 0},
	{0, 0, 70000, 0},
	{0, 0, 1.5, 0},

	//theta 1 zero
	{1, 0, 4, 1},
	{50, 0, 33, 50},
	{700.55, 0, 1122, 700.55},

	//regular
	{5, 10, 2, 25},
	{1, 2, 50, 101},
}

func TestRunHypothesis(t *testing.T) {
	for _, testCase := range hypTests {
		linearModel := LinearModel{trained: true, theta0: testCase.theta0, theta1: testCase.theta1}

		y, err := linearModel.RunHypothesis(testCase.x)
		if err != nil {
			t.Error(err)
		}

		if y != testCase.y {
			t.Error(
				"For:", testCase.x,
				"Expected:", testCase.y,
				"Got:", y,
			)
		}
	}
}

func TestTrain(t *testing.T) {
	//load training data
	trainingData, err := dataset.Load("../testdata/univariate_linear.csv")
	if err != nil {
		t.Error(err)
	}

	//run train
	model := NewLinearModel()
	err = model.Train(*trainingData, .00001)
	if err != nil {
		t.Error(err)
	}

	//TODO: need some type of acceptance conditions
}

func TestMeanSquaredError(t *testing.T) {
	model := LinearModel{trained: true, theta0: 1, theta1: 1}

	dataSet := [][]float64{
		{6, 7},
		{1, 2},
		{0, 1},
	}

	mse, err := model.MeanSquaredError(dataSet)
	if err != nil {
		t.Error(err)
	}

	if mse != 0 {
		t.Error("MSE calculation incorrect")
	}

}
