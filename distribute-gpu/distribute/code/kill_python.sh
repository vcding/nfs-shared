kill -9 `ps -a | grep python | awk {'print $1'}`
