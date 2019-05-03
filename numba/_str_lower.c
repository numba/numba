#include <Python.h>
#include "_pymodule.h"
#include <stdio.h>


/* from https://github.com/python/cpython/blob/235e7b2b0d937ba8b4a9172aa72206187e3e1f54/Objects/unicodectype.c#L13 */
#define LOWER_MASK 0x08
#define UPPER_MASK 0x80
#define EXTENDED_CASE_MASK 0X4000

typedef struct {
	const int upper;
	const int lower;
	const int title;
	const unsigned char decimal;
	const unsigned char digit;
	const unsigned short flags;
} _PyUnicode_TypeRecord;

#include "unicodetype_db.h"

/*from https://github.com/python/cpython/blob/e8113f51a8bdf33188ee30a1c038a298329e7bfa/Objects/unicodectype.c#L45 */
static const _PyUnicode_TypeRecord * gettyperecord(char c)
{
	int index;
	if (c >= 0x110000){
	   index = 0;
    }
	else{
        index = index1[(c>>SHIFT)];
        index = index2[(index<<SHIFT)+(c&((1<<SHIFT)-1))];
	}
    
    return &_PyUnicode_TypeRecords[index];

}

/* from https://github.com/python/cpython/blob/e8113f51a8bdf33188ee30a1c038a298329e7bfa/Objects/unicodectype.c#L211 */
int _PyUnicode_ToLower(char ch, char *res)
{
    const _PyUnicode_TypeRecord *ctype = gettyperecord(ch);

    if(ctype->flags & EXTENDED_CASE_MASK){
        int index = ctype->lower & 0xFFFF;
        int n = ctype->lower >> 24;
        int i;
        for(i = 0; i < n; i++){
            res[i] = _PyUnicode_ExtendedCase[index + i];
        }
        return n;
    } else
    {
        for(unsigned int i=0; i < strlen(res); i++){
            const _PyUnicode_TypeRecord *temptype = gettyperecord(res[i]);
            res[i] = res[i] + temptype->lower;
        }
        return 1;
    }
}

int strlower(char *str)
{
	char ch = str[0];
    return _PyUnicode_ToLower(ch, str);
}

MOD_INIT(string_lower_ext) {
    PyObject *m;
    MOD_DEF(m, "string_lower_ext", "No docs", NULL)
        if (m == NULL)
            return MOD_ERROR_VAL;
    PyModule_AddObject(m, "strlower", PyLong_FromVoidPtr(&strlower));
    return MOD_SUCCESS_VAL(m);
}
