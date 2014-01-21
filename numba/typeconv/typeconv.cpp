#include <cstring>
#include <algorithm>

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

Type TypeManager::get(const char name[]) {
	std::string key(name);
	if (typemap.find(key) == typemap.end()) {
		// not found
		Type ty(typemap.size());
		typemap.insert(std::make_pair(key, ty));
		return ty;
	} else {
		// found
		return typemap[key];
	}
}

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
	tccmap[pair] = tcc;
}

TypeCompatibleCode TypeManager::isCompatible(Type from, Type to) const {
	if (from == to)
		return TCC_EXACT;
	TypePair pair(from, to);
	TCCMap::const_iterator it = tccmap.find(pair);
	if (it == tccmap.end())
		return TCC_FALSE;
	else
		return it->second;
}


int TypeManager::selectOverload(Type sig[], Type ovsigs[], int sigsz, int ovct) const {
	int sel;
	if (ovct < 16) {
		Rating ratings[16];
		sel = _selectOverload(sig, ovsigs, sigsz, ovct, ratings);
	} else if (ovct < 128) {
		Rating ratings[128];
		sel = _selectOverload(sig, ovsigs, sigsz, ovct, ratings);
	} else {
		Rating *ratings = new Rating[ovct];
		sel = _selectOverload(sig, ovsigs, sigsz, ovct, ratings);
		delete [] ratings;
	}
	return sel;
}

int TypeManager::_selectOverload(Type sig[], Type ovsigs[], int sigsz, int ovct,
                                 Rating ratings[]) const {
	// Generate rating table
	// Use a penalize scheme.
	for (int i = 0; i < ovct; ++i) {
		Type* entry = &ovsigs[i * sigsz];

		Rating &rate = ratings[i];
		for (int j = 0; j < sigsz; ++j) {
			TypeCompatibleCode tcc = isCompatible(sig[j], entry[j]);
			if (tcc == TCC_FALSE) {
				rate.bad();
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

	// Find lowest rating
	Rating best;
	best.bad();
	int bestsel = 0;
	int bestrep = 0;
	for (int i = 0; i < ovct; ++i) {
		if (ratings[i] < best){
			best = ratings[i];
			bestrep = 0;
			bestsel = i;
		} else if (ratings[i] == best) {
			bestrep += 1;
		}
	}

	if (bestrep != 0)
		return -1;

	return bestsel;
}

// ----- Ratings -----
Rating::Rating() :promote(0), safe_convert(0), unsafe_convert(0) { }

void Rating::bad() {
    // Max out everything
    promote = -1;
    safe_convert = -1;
    unsafe_convert = -1;
}

bool Rating::operator < (const Rating &other) const {
    unsigned short self[] = {unsafe_convert,
                             safe_convert,
                             promote};
    unsigned short that[] = {other.unsafe_convert,
                             other.safe_convert,
                             other.promote};
    for(int i = 0; i < sizeof(self)/sizeof(unsigned short); ++i) {
        if (self[i] < that[i]) return true;
    }
    return false;
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

