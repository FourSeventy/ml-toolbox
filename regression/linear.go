package regression

import (
	"math"
)

//LinearModel represents a trained linear regression model
type LinearModel struct {
	Theta0 float64
	Theta1 float64
}

//NewLinearModel constructs an untrained LinearModel
func NewLinearModel() *LinearModel {
	model := LinearModel{Theta0: 0, Theta1: 0}
	return &model
}

//Train trains a linear model with the given trainingSet and step size
//TODO: better trainingSet type
func (model *LinearModel) Train(trainingSet [][]float64, step float64) {

	//pick a convergence threshold
	const threshold = .000001

	//perform gradient descent optimization
	oldTheta0 := model.Theta0 + .002
	oldTheta1 := model.Theta1 + .002
	for (math.Abs(oldTheta0-model.Theta0) > threshold) && (math.Abs(oldTheta1-model.Theta1) > threshold) {

		//calculate new theta 0
		var sum float64
		for _, example := range trainingSet {
			//for each of the test set examples, sum the difference of h(x) and y
			sum += model.RunHypothesis(example[0]) - example[1]
		}
		tempTheta0 := model.Theta0 - (step/float64(len(trainingSet)))*sum

		//calculate new theta 1
		sum = 0
		for _, example := range trainingSet {
			//for each of the test set examples, sum the difference of h(x) and y multiplied by x
			sum += (model.RunHypothesis(example[0]) - example[1]) * example[0]
		}
		tempTheta1 := model.Theta1 - (step/float64(len(trainingSet)))*sum

		//update old thetas
		oldTheta0 = model.Theta0
		oldTheta1 = model.Theta1

		//assign new thetas
		model.Theta0 = tempTheta0
		model.Theta1 = tempTheta1

		// fmt.Printf("Theta0: %v, Theta1: %v \n", model.Theta0, model.Theta1)
	}

}

//RunHypothesis runs the hypothesis calculation on a trained LinearModel
func (model LinearModel) RunHypothesis(x float64) float64 {
	y := model.Theta0 + model.Theta1*x
	return y
}
