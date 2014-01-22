#include "typeconv/typeconv.hpp"

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

    Type get_type(const char *name) {
        return tm->get(name);
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
    Dispatcher *disp = static_cast<Dispatcher*>(obj);
    Type *args = new Type[disp->argct];
    for(int i = 0; i < disp->argct; ++i) {
        args[i] = Type(tys[i]);
    }

    disp->addDefinition(args, callable);
    delete [] args;
}

void*
dispatcher_resolve(void* obj, int sig[]) {
    Type prealloc[12];
    Dispatcher *disp = static_cast<Dispatcher*>(obj);
    Type *args;
    if (disp->argct < sizeof(prealloc) / sizeof(Type))
        args = prealloc;
    else
        args = new Type[disp->argct];

    for(int i = 0; i < disp->argct; ++i) {
        args[i] = Type(sig[i]);
    }

    void *callable = disp->resolve(args);

    if (args != prealloc)
        delete [] args;
    return callable;
}

int
dispatcher_get_type(void *obj, const char *name) {
    Dispatcher *disp = static_cast<Dispatcher*>(obj);
    return disp->get_type(name).get();
}
