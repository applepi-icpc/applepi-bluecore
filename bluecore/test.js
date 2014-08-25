var pkucaptcha = require('./build/Release/pkucaptcha');

console.log('ANS =', pkucaptcha.identify('./sample/DrawServlet20.jpeg'));

