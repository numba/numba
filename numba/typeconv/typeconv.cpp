#include <cstring>
#include <cstdio>
#include <algorithm>
#include <limits.h>

#include "typeconv.hpp"


// ------ TypeManager ------

TCCMap::TCCMap()
    : nb_records(0)
{
}

unsigned int TCCMap::hash(const TypePair &key) const {
    const int mult = 1000003;
    int x = 0x345678;
    x = (x ^ key.first) * mult;
    x = (x ^ key.second);
    return x;
}

void TCCMap::insert(const TypePair &key, TypeCompatibleCode val) {
    unsigned int i = hash(key) & (TCCMAP_SIZE - 1);
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
    nb_records++;
}

TypeCompatibleCode TCCMap::find(const TypePair &key) const {
    unsigned int i = hash(key) & (TCCMAP_SIZE - 1);
    const TCCMapBin &bin = records[i];
    for (unsigned int j = 0; j < bin.size(); ++j) {
        if (bin[j].key == key) {
            return bin[j].val;
        }
    }
    return TCC_FALSE;
}

// ----- Ratings -----
Rating::Rating() : promote(0), safe_convert(0), unsafe_convert(0) { }

inline bool Rating::operator < (const Rating &other) const {
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

inline bool Rating::operator == (const Rating &other) const {
    return promote == other.promote && safe_convert == other.safe_convert &&
           unsafe_convert == other.unsafe_convert;
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


int TypeManager::selectOverload(const Type sig[], const Type ovsigs[],
                                int &selected,
                                int sigsz, int ovct, bool allow_unsafe) const {
    int count;
    if (ovct <= 16) {
        Rating ratings[16];
        int candidates[16];
        count = _selectOverload(sig, ovsigs, selected, sigsz, ovct,
                                allow_unsafe, ratings, candidates);
    }
    else {
        Rating *ratings = new Rating[ovct];
        int *candidates = new int[ovct];
        count = _selectOverload(sig, ovsigs, selected, sigsz, ovct,
                                allow_unsafe, ratings, candidates);
        delete [] ratings;
        delete [] candidates;
    }
    return count;
}

int TypeManager::_selectOverload(const Type sig[], const Type ovsigs[],
                                 int &selected, int sigsz, int ovct,
                                 bool allow_unsafe, Rating ratings[],
                                 int candidates[]) const {
    // Generate rating table
    // Use a penalize scheme.
    int nb_candidates = 0;

    for (int i = 0; i < ovct; ++i) {
        const Type *entry = &ovsigs[i * sigsz];
        Rating rate;

        for (int j = 0; j < sigsz; ++j) {
            TypeCompatibleCode tcc = isCompatible(sig[j], entry[j]);
            if (tcc == TCC_FALSE ||
                (tcc == TCC_CONVERT_UNSAFE && !allow_unsafe)) {
                // stop the loop early
                goto _incompatible;
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
        ratings[nb_candidates] = rate;
        candidates[nb_candidates] = i;
        nb_candidates++;
    _incompatible:
        ;
    }

    // Bail if no match
    if (nb_candidates == 0)
        return 0;

    // Find lowest rating
    Rating best = ratings[0];
    selected = candidates[0];

    int matchcount = 1;
    for (int i = 1; i < nb_candidates; ++i) {
        if (ratings[i] < best) {
            best = ratings[i];
            selected = candidates[i];
            matchcount = 1;
        }
        else if (ratings[i] == best) {
            matchcount += 1;
        }
    }
    return matchcount;
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

