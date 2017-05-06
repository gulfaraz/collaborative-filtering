var mathjs = require("mathjs");

var randnBoxMuller = require("ml-util/randnBoxMuller");

var computeNumericalGradient
    = require("./computeNumericalGradient");
var collaborativeFilteringCostFunction
    = require("./collaborativeFilteringCostFunction");

module.exports = function (lambda) {

    if(!lambda) {
        lambda = 0;
    }

    var X_t = mathjs.matrix(mathjs.random([ 4, 3 ]));
    var Theta_t = mathjs.matrix(mathjs.random([ 5, 3 ]));

    var Y = mathjs.multiply(X_t, mathjs.transpose(Theta_t));

    var R = mathjs.matrix(mathjs.randomInt(Y._size, 0, 2));

    Y = mathjs.dotMultiply(Y, R);

    var X = mathjs.matrix(mathjs.random(X_t._size));

    mathjs.map(X, function (value, index, matrix) {
        X = mathjs.subset(
            X,
            mathjs.index(index[0], index[1]),
            randnBoxMuller()
        );
    });

    var Theta = mathjs.matrix(mathjs.random(Theta_t._size));

    mathjs.map(Theta_t, function (value, index, matrix) {
        Theta_t = mathjs.subset(
            Theta_t,
            mathjs.index(index[0], index[1]),
            randnBoxMuller()
        );
    });

    var numberOfUsers = Y._size[1];
    var numberOfMovies = Y._size[0];
    var numberOfFeatures = Theta_t._size[1];

    var parameters = mathjs.concat(
        mathjs.reshape(
            X,
            [ X._size[0] * X._size[1], 1 ]
        ),
        mathjs.reshape(
            Theta,
            [ Theta._size[0] * Theta._size[1], 1 ]
        ),
        0
    );

    var numgrad = computeNumericalGradient(function (t) {
        return collaborativeFilteringCostFunction(
            t,
            Y,
            R,
            numberOfUsers,
            numberOfMovies,
            numberOfFeatures,
            lambda
        );
    }, parameters);

    var [ cost, grad ] = collaborativeFilteringCostFunction(
        parameters,
        Y,
        R,
        numberOfUsers,
        numberOfMovies,
        numberOfFeatures,
        lambda
    );

    grad = mathjs.reshape(grad, grad._size.concat(1));

    mathjs.forEach(numgrad, function (value, index, matrix) {
        console.log(
            mathjs.subset(numgrad, mathjs.index.apply(null, index)),
            mathjs.subset(grad, mathjs.index.apply(null, index))
        );
    });

    console.log("\nThe above two columns you get should be similar -"
        + " (Left-Your Numerical Gradient"
        + ", Right-Analytical Gradient)");

    var diff = mathjs.divide(
        mathjs.norm(mathjs.flatten(mathjs.subtract(numgrad, grad))),
        mathjs.norm(mathjs.flatten(mathjs.add(numgrad, grad)))
    );

    console.log("\nIf the cost function implementation is correct,"
        + " then the relative difference will be small"
        + " (less than 1e-9)\nRelative Difference: " + diff + "\n");
};
