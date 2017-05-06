var mathjs = require("mathjs");

module.exports = function (J, theta) {

    var numgrad = mathjs.zeros.apply(null, theta._size);
    var perturb = mathjs.zeros.apply(null, theta._size);
    var e = 0.0001;

    mathjs.forEach(numgrad, function(value, index, matrix) {
        perturb = mathjs.subset(
            perturb,
            mathjs.index.apply(null, index),
            e
        );

        var loss1 = J(mathjs.subtract(theta, perturb))[0];
        var loss2 = J(mathjs.add(theta, perturb))[0];
        numgrad = mathjs.subset(
            numgrad,
            mathjs.index.apply(null, index),
            ((loss2 - loss1)/(2*e))
        );

        perturb = mathjs.subset(
            perturb,
            mathjs.index.apply(null, index),
            0
        );
    });

    return numgrad;
};
