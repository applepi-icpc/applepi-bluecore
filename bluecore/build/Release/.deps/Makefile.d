cmd_Makefile := cd ..; /usr/local/lib/node_modules/node-gyp/gyp/gyp_main.py -fmake --ignore-environment "--toplevel-dir=." -I/Users/applepi/Documents/project-watermetre/build/config.gypi -I/usr/local/lib/node_modules/node-gyp/addon.gypi -I/Users/applepi/.node-gyp/0.10.31/common.gypi "--depth=." "-Goutput_dir=." "--generator-output=build" "-Dlibrary=shared_library" "-Dvisibility=default" "-Dnode_root_dir=/Users/applepi/.node-gyp/0.10.31" "-Dmodule_root_dir=/Users/applepi/Documents/project-watermetre" binding.gyp