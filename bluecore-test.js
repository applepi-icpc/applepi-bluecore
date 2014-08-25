var identifier = require('./identify.js');
var logger = require('./login.js');

var infinitelyTry = function (jsessionid) {
	identifier.tryIdentify(jsessionid, function (err, success) {
		if (err) {
			console.log(err);
		}
		else if (success) {
			console.log("\033[32m\033[1mSUCCESS\033[0m");
		}
		else {
			console.log("\033[31m\033[1mFAILED\033[0m");
		}
		if (identifier.tried % 10 == 0) {
			console.log('\033[34mTried ' + identifier.tried + ', correct ' + identifier.correct + '. Accuracy ' + identifier.correct * 100 / identifier.tried + '%.\033[0m');
		}
		setTimeout(function () {
			infinitelyTry(jsessionid);
		}, 2000);
	});
};

logger.login('1200012716', 'WoShiRen2', function (err, jsessionid) {
	if (err) {
		console.log(err);
	}
	else {
		console.log("LOGGED IN.");
		console.log("\033[34m-- BEGIN TO TEST BLUECORE --\033[0m");
		infinitelyTry(jsessionid);
	}
});

