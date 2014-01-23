#include "typeconv/typeconv.hpp"
#include <cassert>
#include <vector>

typedef std::vector<Type> TypeTable;
typedef std::vector<void*> Functions;

class Dispatcher {
public:
    Dispatcher(TypeManager *tm, int argct): argct(argct), tm(tm), ovct(0) { }

    void addDefinition(Type args[], void *callable) {
        overloads.reserve(argct + overloads.size());
        for (int i=0; i<argct; ++i) {
            overloads.push_back(args[i]);
        }
        functions.push_back(callable);
    }

    void* resolve(Type sig[]) {
        int sel = tm->selectOverload(sig, &overloads[0], argct,
                                     functions.size());

        if (sel == -1) {
            return NULL;
        }

        return functions[sel];
    }

    const int argct;
private:
    TypeManager *tm;
    TypeTable overloads;
    Functions functions;
    int ovct;
};


#include "_dispatcher.h"

void*
dispatcher_new(void *tm, int argct){
    return new Dispatcher(static_cast<TypeManager*>(tm), argct);
}

void
dispatcher_del(void *obj) {
    Dispatcher *disp = static_cast<Dispatcher*>(obj);
    delete disp;
}

void
dispatcher_add_defn(void *obj, int tys[], void* callable) {
    assert(sizeof(int) == sizeof(Type) &&
            "Type should be representable by an int");

    Dispatcher *disp = static_cast<Dispatcher*>(obj);
    Type *args = reinterpret_cast<Type*>(tys);
    disp->addDefinition(args, callable);
}

void*
dispatcher_resolve(void* obj, int sig[]) {
    Dispatcher *disp = static_cast<Dispatcher*>(obj);
    Type *args = reinterpret_cast<Type*>(sig);
    void *callable = disp->resolve(args);
    return callable;
}
