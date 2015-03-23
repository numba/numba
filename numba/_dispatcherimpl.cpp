#include "typeconv/typeconv.hpp"
#include <cassert>
#include <vector>

typedef std::vector<Type> TypeTable;
typedef std::vector<void*> Functions;

struct _opaque_dispatcher {};

class Dispatcher: public _opaque_dispatcher {
public:
    Dispatcher(TypeManager *tm, int argct): argct(argct), tm(tm) { }

    void addDefinition(Type args[], void *callable) {
        overloads.reserve(argct + overloads.size());
        for (int i=0; i<argct; ++i) {
            overloads.push_back(args[i]);
        }
        functions.push_back(callable);
    }

    void* resolve(Type sig[], int &matches, bool allow_unsafe) {
        const int ovct = functions.size();
        int selected;
        matches = 0;
        if (0 == ovct) {
            return NULL;
        }
        if (overloads.size() > 0) {
            matches = tm->selectOverload(sig, &overloads[0], selected, argct,
                                         ovct, allow_unsafe);
        } else if (argct == 0){
            matches = 1;
            selected = 0;
        }
        if (matches == 1){
            return functions[selected];
        }
        return NULL;
    }

    int count() const { return functions.size(); }

    void clear() {
        functions.clear();
        overloads.clear();
    }

private:
    const int argct;
    TypeManager *tm;
    TypeTable overloads;
    Functions functions;
};


#include "_dispatcher.h"

dispatcher_t *
dispatcher_new(void *tm, int argct){
    return new Dispatcher(static_cast<TypeManager*>(tm), argct);
}

void
dispatcher_clear(dispatcher_t *obj) {
    Dispatcher *disp = static_cast<Dispatcher*>(obj);
    disp->clear();
}

void
dispatcher_del(dispatcher_t *obj) {
    Dispatcher *disp = static_cast<Dispatcher*>(obj);
    delete disp;
}

void
dispatcher_add_defn(dispatcher_t *obj, int tys[], void* callable) {
    assert(sizeof(int) == sizeof(Type) &&
            "Type should be representable by an int");

    Dispatcher *disp = static_cast<Dispatcher*>(obj);
    Type *args = reinterpret_cast<Type*>(tys);
    disp->addDefinition(args, callable);
}

void*
dispatcher_resolve(dispatcher_t *obj, int sig[], int *count, int allow_unsafe) {
    Dispatcher *disp = static_cast<Dispatcher*>(obj);
    Type *args = reinterpret_cast<Type*>(sig);
    void *callable = disp->resolve(args, *count, (bool) allow_unsafe);
    return callable;
}

int
dispatcher_count(dispatcher_t *obj) {
    Dispatcher *disp = static_cast<Dispatcher*>(obj);
    return disp->count();
}
