var http = require('http');
var https = require('https');
var querystring = require('querystring');

var timeout = 10000;

var exports = [];
module.exports = exports;

// callback(error, jsessionid)
exports.login = function (username, password, callback) {
	var reqSsoLogin = https.request({
		hostname: 'iaaa.pku.edu.cn',
		path: '/iaaa/oauth.jsp?appID=syllabus&appName=学生选课系统&redirectUrl=http://elective.pku.edu.cn:80/elective2008/agent4Iaaa.jsp/../ssoLogin.do',
		method: 'GET'
	}, function (response) {
		var jsessionid = response.headers['set-cookie'][0].match(/JSESSIONID=([^;]+)/)[1];
		reqSsoLogin.abort();
		var reqOAuth = https.request({
			hostname: 'iaaa.pku.edu.cn',
			path: '/iaaa/oauthlogin.do',
			headers: {
				'cookie': 'JSESSIONID=' + jsessionid,
				'content-type': 'application/x-www-form-urlencoded'
			},
			method: 'POST'
		}, function (response) {
			var buffers = [];
			var len = 0;
			response.on('data', function (chunk) {
				buffers.push(chunk);
				len += chunk.length;
			});
			response.on('end', function () {
				var resBody = Buffer.concat(buffers, len).toString('utf8');
				if (resBody.search(/"success":\s*true/) == -1) {
					var err = resBody.match(/"msg":\s*"([^"]*)"/)[1];
					return callback(err);
				}
				var token = resBody.match(/"token":\s*"(\w+)"/)[1];
				var reqElec = http.request({
					hostname: 'elective.pku.edu.cn',
					path: '/elective2008/ssoLogin.do?token=' + token,
					method: 'GET'
				}, function (response) {			
					var jsessionid = response.headers['set-cookie'][0].match(/JSESSIONID=([^;]+)/)[1];
					reqElec.abort();
					return callback(undefined, jsessionid);
				});
				reqElec.on('socket', function (socket) {
					socket.setTimeout(timeout);
					socket.on('timeout', function () {
						reqElec.abort();
					});
				});
				reqElec.on('error', function(err) {
					return callback(err);
				});
				reqElec.end();
			});
		});
		reqOAuth.write(querystring.stringify({
			appid: 'syllabus',
			userName: username,
			password: password,
			redirUrl: 'http://elective.pku.edu.cn:80/elective2008/agent4Iaaa.jsp/../ssoLogin.do'
		}), 'utf8');
		reqOAuth.on('socket', function (socket) {
			socket.setTimeout(timeout);
			socket.on('timeout', function () {
				reqOAuth.abort();
			});
		});
		reqOAuth.on('error', function(err) {
			return callback(err);
		});
		reqOAuth.end();
	});
	reqSsoLogin.on('socket', function (socket) {
		socket.setTimeout(timeout);
		socket.on('timeout', function () {
			reqSsoLogin.abort();
		});
	});
	reqSsoLogin.on('error', function(err) {
		return callback(err);
	});
	reqSsoLogin.end();
};

