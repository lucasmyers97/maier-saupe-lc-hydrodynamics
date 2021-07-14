#include <iostream>
#include <math.h>

int main()
{
  int count = 30;

  double phi{0};
  double x{0};
  double y{0};
  double z{0};
  for (int i = 0; i < count; i++) {
    phi = i * 4*M_PI / count;
    x = cos(phi);
    y = sin(phi);
    z = atan2(y, x) + 2.0*M_PI;
    z = fmod(z, 2.0*M_PI);

    std::cout << z << std::endl;
  }

  return 0;
}
