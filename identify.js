var bluecore = require('./bluecore/pkucaptcha');
var http = require('http');
var fs = require('fs');

var timeout = 10000; // milliseconds

var exports = [];
module.exports = exports;

exports.tried = 0;
exports.correct = 0;
exports.wrong = 0;

var errId = 1001;

var id = 0;
var getUid = function () {
	++id;
	if (id == 100000) id = 0;
	return '' + new Date().getTime() + id;
}

// callback(err, success)
exports.tryIdentify = function (jsessionid, callback) {
	var reqCaptcha = http.request({
		hostname: 'elective.pku.edu.cn',
		path: '/elective2008/DrawServlet',
		headers: {
			'cookie': 'JSESSIONID=' + jsessionid
		},
		method: 'GET'
	}, function (response) {
		var buffers = [];
		var lenRaw = 0;
		response.on('data', function (chunk) {
			buffers.push(chunk);
			lenRaw += chunk.length;
		});
		response.on('end', function () {
			var raw = Buffer.concat(buffers, lenRaw);
			var filePath = './tmp/' + getUid() + '.jpg';
			var filePathErr = './bluecore/training/realdata/cap' + errId + '.jpg';
			fs.open(filePath, 'wx', function (err, fd) {
				if (err) {
					return callback(err, false);
				}
				fs.write(fd, raw, 0, lenRaw, null, function (err) {
					if (err) {
						return callback(err, false);
					}
					fs.close(fd, function () {
						var res = bluecore.identify(filePath);
						console.log(res);
						fs.unlink(filePath, function () {
							var reqSubmit = http.request({
								hostname: 'elective.pku.edu.cn',
								path: '/elective2008/edu/pku/stu/elective/controller/supplement/validate.do?validCode=' + res,
								headers: {
									'cookie': 'JSESSIONID=' + jsessionid
								},
								method: 'GET'
							}, function (response) {
								var buffers = [];
								var len = 0;
								response.on('data', function (chunk) {
									buffers.push(chunk);
									len += chunk.length;
								});
								response.on('end', function () {
									var resBody = Buffer.concat(buffers, len).toString('utf8');
									++exports.tried;
									if (resBody.search('<valid>2</valid>') == -1) {
										++exports.wrong;
										fs.open(filePathErr, 'w', function (err, fdErr) {
											if (!err) {
												fs.write(fdErr, raw, 0, lenRaw, null, function (err) {
													if (!err) {
														fs.close(fdErr, function () {
															++errId;
															console.log('The CAPTCHA which was incorrectly identified has been saved.');
														});
													}
												});
											}
										});
										return callback(undefined, false);
									}
									else {
										++exports.correct;
										return callback(undefined, true);
									}
								});
							});
							reqSubmit.on('socket', function (socket) {
								socket.setTimeout(timeout);
								socket.on('timeout', function () {
									reqSubmit.abort();
								});
							});
							reqSubmit.on('error', function(err) {
								return callback(err);
							});
							reqSubmit.end();
						});
					});
				});
			});
		});
	});
	reqCaptcha.on('socket', function (socket) {
		socket.setTimeout(timeout);
		socket.on('timeout', function () {
			reqCaptcha.abort();
		});
	});
	reqCaptcha.on('error', function(err) {
		return callback(err);
	});
	reqCaptcha.end();
};
