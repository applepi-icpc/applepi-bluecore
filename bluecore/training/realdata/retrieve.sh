#!/bin/bash

for i in {1..1000}; do
	echo "curl http://elective.pku.edu.cn/elective2008/DrawServlet > cap$i.jpg" | sh
done

