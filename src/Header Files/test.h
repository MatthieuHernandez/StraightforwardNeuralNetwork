#pragma once
#pragma warning(push, 0) 
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#pragma warning(pop)

class Test
{
public :

	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, unsigned version)
	{
		ar & x;
	}

public:

	Test(int x)
	{
		this->x = x;
	}

	Test() = default;
	~Test() = default;

	virtual void tutu() = 0;


	int x;
};

class Test2 : public Test
{
public :

	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive& ar, unsigned version)
	{
		boost::serialization::void_cast_register<Test2, Test>();
		//boost::serialization::base_object<Test>(*this);
		ar & x;
		ar & y;
	}

public:

	void tutu() override {};

	Test2(int x, int y)
		: Test(x)
	{
		this->y = y;
	}

	Test2() = default;
	~Test2() = default;


	int y;
};
