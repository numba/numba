#include <cstring>
#include <algorithm>
#include <limits.h>
#include "typeconv.hpp"


// ----- Type -----

Type::Type() :id(-1) { }

Type::Type(int id) :id(id) { } 

Type::Type(const Type& other) :id(other.id) { }

Type& Type::operator = (const Type& other) {
    id = other.id;
    return *this;
}

bool Type::valid() const {
    return id != -1;
}

bool Type::operator == (const Type& other) const {
    return id == other.id;
}

bool Type::operator != (const Type& other) const {
    return id != other.id;
}

bool Type:: operator < (const Type& other) const {
    return id < other.id;
}

int Type::get() const { return id; }

// ------ TypeManager ------

unsigned int TCCMap::hash(TypePair key) const {
    int a = key.first.get() * 9973;
    int b = key.second.get() * 10007;

    a += b;
    b = (b << 17) | (b >> (32 - 17));
    return b ^ a;
}

void TCCMap::insert(TypePair key, TypeCompatibleCode val) {
    unsigned int i = hash(key) % TCCMAP_SIZE;
    TCCMapBin &bin = records[i];
    TCCRecord data;
    data.key = key;
    data.val = val;
    for (unsigned int j = 0; j < bin.size(); ++j) {
        if (bin[j].key == key) {
            bin[j].val = val;
            return;
        }
    }
    bin.push_back(data);
}

TypeCompatibleCode TCCMap::find(TypePair key) const {
    unsigned int i = hash(key) % TCCMAP_SIZE;
    const TCCMapBin &bin = records[i];
    for (unsigned int j = 0; j < bin.size(); ++j) {
        if (bin[j].key == key) {
            return bin[j].val;
        }
    }
    return TCC_FALSE;
}

// ------ TypeManager ------

bool TypeManager::canPromote(Type from, Type to) const {
    return isCompatible(from, to) == TCC_PROMOTE;
}

bool TypeManager::canSafeConvert(Type from, Type to) const {
    return isCompatible(from, to) == TCC_CONVERT_SAFE;
}

bool TypeManager::canUnsafeConvert(Type from, Type to) const {
    return isCompatible(from, to) == TCC_CONVERT_UNSAFE;
}

void TypeManager::addPromotion(Type from, Type to) {
    return addCompatibility(from, to, TCC_PROMOTE);
}

void TypeManager::addUnsafeConversion(Type from, Type to) {
    return addCompatibility(from, to, TCC_CONVERT_UNSAFE);
}

void TypeManager::addSafeConversion(Type from, Type to) {
    return addCompatibility(from, to, TCC_CONVERT_SAFE);
}

void TypeManager::addCompatibility(Type from, Type to, TypeCompatibleCode tcc) {
    TypePair pair(from, to);
    tccmap.insert(pair, tcc);
}

TypeCompatibleCode TypeManager::isCompatible(Type from, Type to) const {
    if (from == to)
        return TCC_EXACT;
    TypePair pair(from, to);
    return tccmap.find(pair);
}


int TypeManager::selectOverload(Type sig[], Type ovsigs[], int &selected,
                                int sigsz, int ovct, bool allow_unsafe) const {
    int count;
    if (ovct <= 16) {
        Rating ratings[16];
        count = _selectOverload(sig, ovsigs, selected, sigsz, ovct,
                                allow_unsafe, ratings);
    }
    else {
        Rating *ratings = new Rating[ovct];
        count = _selectOverload(sig, ovsigs, selected, sigsz, ovct,
                                allow_unsafe, ratings);
        delete [] ratings;
    }
    return count;
}

int TypeManager::_selectOverload(Type sig[], Type ovsigs[], int &selected,
                                 int sigsz, int ovct, bool allow_unsafe,
                                 Rating ratings[]) const {
    // Generate rating table
    // Use a penalize scheme.
    int badcount = 0;
    for (int i = 0; i < ovct; ++i) {
        Type* entry = &ovsigs[i * sigsz];

        Rating &rate = ratings[i];
        for (int j = 0; j < sigsz; ++j) {
            TypeCompatibleCode tcc = isCompatible(sig[j], entry[j]);
            if (tcc == TCC_FALSE ||
                (tcc == TCC_CONVERT_UNSAFE && !allow_unsafe)) {
                rate.bad();
                ++badcount;
                break; // stop the loop early for incompatbile type
            }
            switch(tcc) {
            case TCC_PROMOTE:
                rate.promote += 1;
                break;
            case TCC_CONVERT_SAFE:
                rate.safe_convert += 1;
                break;
            case TCC_CONVERT_UNSAFE:
                rate.unsafe_convert += 1;
                break;
            default:
                break;
            }
        }
    }

    // Fast path for no match
    if (badcount == ovct)
        return 0;

    // Find lowest rating
    Rating best;
    best.bad();

    int matchcount = 0;
    for (int i = 0; i < ovct; ++i) {
        if (ratings[i] < best){
            best = ratings[i];
            matchcount = 1;
            selected = i;
        }
        else if (ratings[i] == best) {
            matchcount += 1;
        }
    }
    return matchcount;
}

// ----- Ratings -----
Rating::Rating() :promote(0), safe_convert(0), unsafe_convert(0) { }

void Rating::bad() {
    // Max out everything
    promote = UINT_MAX;
    safe_convert = UINT_MAX;
    unsafe_convert = UINT_MAX;
}

bool Rating::operator < (const Rating &other) const {
    if (unsafe_convert < other.unsafe_convert)
        return true;
    else if (unsafe_convert > other.unsafe_convert)
        return false;
    if (safe_convert < other.safe_convert)
        return true;
    else if (safe_convert > other.safe_convert)
        return false;
    return (promote < other.promote);
}

bool Rating::operator == (const Rating &other) const {
    return promote == other.promote && safe_convert == other.safe_convert &&
           unsafe_convert == other.unsafe_convert;
}


// ----- utils -----

const char* TCCString(TypeCompatibleCode tcc) {
    switch(tcc) {
    case TCC_EXACT:
        return "exact";
    case TCC_SUBTYPE:
        return "subtype";
    case TCC_PROMOTE:
        return "promote";
    case TCC_CONVERT_SAFE:
        return "safe_convert";
    case TCC_CONVERT_UNSAFE:
        return "unsafe_convert";
    default:
        return "false";
    }
}

