var mathjs = require("mathjs");

module.exports = function (
    parameters,
    Y,
    R,
    numberOfUsers,
    numberOfMovies,
    numberOfFeatures,
    lambda
) {

    var J = 0;
    var grad = [];

    parameters = mathjs.matrix(parameters);

    var parametersSize = parameters._size;
    var parametersLength
        = parametersSize[0] * parametersSize[1];
    var indexRange1
        = mathjs.range(0, (numberOfMovies * numberOfFeatures));
    var indexRange2
        = mathjs.range(
            (numberOfMovies * numberOfFeatures),
            (numberOfUsers * numberOfFeatures)
            + (numberOfMovies * numberOfFeatures)
        );

    var X = mathjs.subset(parameters, mathjs.index(indexRange1, 0));
    X = mathjs.reshape(X, [ numberOfMovies, numberOfFeatures ]);

    var Theta = mathjs.subset(
        parameters,
        mathjs.index(indexRange2, 0)
    );
    Theta = mathjs.reshape(
        Theta,
        [ numberOfUsers, numberOfFeatures ]
    );

    var diff = mathjs.subtract(
        mathjs.multiply(X, mathjs.transpose(Theta)),
        Y
    );

    var sqdiff = mathjs.dotPow(diff, 2);

    J = (1/2) * mathjs.sum(mathjs.dotMultiply(R, sqdiff));

    var diffDotMultiplyR = mathjs.dotMultiply(diff, R);

    var X_grad = mathjs.multiply(diffDotMultiplyR, Theta);

    var Theta_grad = mathjs.multiply(
        mathjs.transpose(diffDotMultiplyR),
        X
    );

    J = J
        + (
            ((lambda/2) * mathjs.sum(mathjs.dotPow(Theta, 2)))
            + ((lambda/2) * mathjs.sum(mathjs.dotPow(X, 2)))
        );

    X_grad = mathjs.add(X_grad, mathjs.multiply(lambda, X));

    Theta_grad = mathjs.add(
        Theta_grad,
        mathjs.multiply(lambda, Theta)
    );

    var X_grad_flat = mathjs.flatten(X_grad);

    var Theta_grad_flat = mathjs.flatten(Theta_grad);

    grad = mathjs.concat(X_grad_flat, Theta_grad_flat);

    return [ J, grad ];
};
