{
	'targets': [
		{
			'target_name': 'pkucaptcha',
			'sources': [ 'main.cpp' ],
			'include_dirs': [ '/usr/local/include/' ],
			'libraries': ['/usr/local/lib/libjpeg.a'],
			'cflags_cc': ['-O2']
		}
	]
}
