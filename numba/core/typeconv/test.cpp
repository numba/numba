#include <iostream>
#include "typeconv.hpp"

using std::cout;
const char EOL = '\n';

int main() {
	TypeManager tm;
	Type t_int32 = tm.get("int32");
	Type t_float = tm.get("float");
	Type t_int64 = tm.get("int64");

	tm.addConversion(t_int32, t_float);
	tm.addConversion(t_float, t_int32);
	tm.addConversion(t_float, t_int64);
	tm.addPromotion(t_int32, t_int64);

	cout << "int32 -> float " 
		 << TCCString(tm.isCompatible(tm.get("int32"), tm.get("float"))) 
		 << EOL;
	cout << "int32 -> int64 " 
	     << TCCString(tm.isCompatible(tm.get("int32"), tm.get("int64")))
	     << EOL;

	Type sig[] = {t_int32, t_float};
	Type ovsigs[] = {
		t_float, t_float,
		t_int64, t_int64,
		t_int32, t_float,
	};

	int sel = tm.selectOverload(sig, ovsigs, 2, 3);

	cout << "Selected " << sel << '\n';


	
	return 0;
}
