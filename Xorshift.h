#ifndef XORSHIFT_H_INCLUDED
#define XORSHIFT_H_INCLUDED

class Xorshift
{
public:


    unsigned x, y, z, w;

    Xorshift(): x(28173), y(912711293), z(827321), w(8712328) {}
    unsigned next()
    {
        unsigned t = x ^ (x << 11);
        x = y;
        y = z;
        z = w;
        return w = w ^ (w >> 19 ) ^ t ^ (t >> 8);
    }

    double rand_01()
    {
        return next()%1000000001 / 1000000000.0;
    }

    double rand_abs1()
    {
        double x = rand_01();
        if(next() & 1)
            return x;
        return -x;
    }
};

#endif // XORSHIFT_H_INCLUDED
