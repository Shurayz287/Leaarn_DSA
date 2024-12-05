/*
    Learn from G4G, some solve from me
    https://www.geeksforgeeks.org/mathematical-algorithms-difficulty-wise/
*/
# include <bits/stdc++.h>

using namespace std;

class Basic{
    public:

    long long Sum(int n){
        if(n==1) return 1;
        return n+ Sum(n-1);
    }

    long long SumSquares(int n){
        if(n==1) return 1;
        return n*n + SumSquares(n-1);
    }

    long long NthArithmeticProgression(int n1, int d, int N){
        if(N==1) return n1;

        return d+NthArithmeticProgression(n1,d,N-1);
        return n1 + d*(N-1);
    }
    
    long long NthGeometricProgression(int n1, int q, int N){
        if(N==1) return n1;

        return q*NthGeometricProgression(n1,q,N-1);
        return n1*pow(q,N-1);
    }

    long long NthTriangularNumber(int n){
        if(n==1) return 1;

        return n+NthTriangularNumber(n-1);
        return n*(n+1)/2;
    }

    long long SumOfSums(int n){
        if(n==1) return 1;
        return n*(n+1)/2+ SumOfSums(n-1);
    }

    int CountDigits(int n){
        int count=0;

        while(n){
            count++;
            n/=10;
        }

        return count;
    }

    int SumOfDigits(int n){
        int result=0;

        while(n){
            result+=n%10;
            n/=10;
        }

        return result;
    }

    int ReverseDigits(int n){
        int re=0;

        while(n){
            re = re*10 + n%10;
            n/=10;
        }

        return re;
    }

    bool PalindromeNumber(int n){
        int pa=0;

        while(n>pa){
            pa = pa*10+n%10;
            n/=10;
        }

        return pa/10 == n || pa == n;
    }

    int gcdTwoNums1(int num1, int num2){
        if(num1<num2) return gcdTwoNums1(num2,num1);

        if(num1%num2 == 0) return num2;

        else{
            num1 = num1%num2;

            return gcdTwoNums1(num1,num2);
        }
    }
    int gcdTwoNums2(int num1, int num2){
        
        while( min(num1, num2)){

            if(num1>num2) num1 %=num2;
            else num2 %=num1;
        }

        return max(num1, num2);
    }

    int lcmTwoNums(int num1, int num2){
        return num1*num2/gcdTwoNums1(num1,num2);
    }

    bool CheckPrime(int n){
        if(n<2) return false;

        if(n ==2 || n == 3) return true;

        if(n%2==0) return false;

        for(int i=3; i*i<=n; i++)
            if(n%i==0) return false;

        return true;
    }

    long long Factorial(int n){
        if(n==1) return 1;
        return n*Factorial(n-1);
    }

    int gcdMoreNums(vector<int> &nums){
        if(nums.size() == 1) return nums[0];

        int gcd = nums[0];

        for(int i=1; i<nums.size(); i++)
            gcd = gcdTwoNums1(gcd, nums[i]);
        
        return gcd;
    }

    int lcmMoreNums(vector<int> &nums){
        if(nums.size() == 1) return nums[0];

        int lcm = nums[0];

        for(int i=1; i<nums.size(); i++)
            lcm = lcmTwoNums(lcm, nums[i]);
        
        return lcm;
    }

    long long PadovanSequence(int n){
        if(n <=2) return 1;

        return PadovanSequence(n-2) + PadovanSequence(n-3);
    }
};

class Easy{
    public:

    int DigitalRoot(int n){
        if(n<10) return n;
        int sumNum=0;

        while(n){
            sumNum+= n%10;
            n/=10;
        }

        return DigitalRoot(sumNum);
    }

    long long NthFibonacciNumber1(int n){
        if(n<=1) return n;

        return NthFibonacciNumber1(n-1) + NthFibonacciNumber1(n-2);
    }
    long long NthFibonacciNumber2(int n){
        if(n<=1) return n;

        vector<int> fi(n+1);

        fi[0] = 0; fi[1] = 1;

        for(int i=2; i<= n; i++)
            fi[i] = fi[i-1] + fi[i-2];
        
        return fi[n];
    }
    long long NthFibonacciGoldenratio(int n){
        return (int)((pow((1+sqrt(5))*1.0/2, n) + pow((-1+sqrt(5))*1.0/2,n))/sqrt(5));
    }

    bool isPrime(int n){
        if( n<= 1) return false;

        if(n<=3) return true;

        if( n%2==0|| n%3==0) return false;

        for(int i=5; i*i<=n; i+=6)
            if( n%i== 0 || n%(i+2) == 0) return false;

        return true;
    }
    bool ThreeDisctFactor(int n){
        return (int) sqrt(n)*sqrt(n) == n && isPrime((int)sqrt(n)); 
    }

    bool isSquare(int n){
        return (int) sqrt(n)*sqrt(n) == n;
    }
    vector<int> ThreeDivisors(int n){
        vector<int> result;

        for(int i=1; i<=n;i++){

            if(isSquare(i) && isPrime((int)sqrt(i))) 
                result.push_back(i);
        }

        return result;
    }

    float SquareRoot(int n){
        if(n==1) return 1;

        float left = 0,
              right=n;
        float mid=0;

        while(abs(mid*mid-n)>=0.00001){
            
            mid = (left+right)*1.0/2;
            
            if(mid*mid == n) return mid;

            if(mid*mid > n) right = mid;
            
            else left = mid;
        }

        return mid;
    }

    vector<vector<int>> PascalTriangle(int n){
        vector<vector<int>> result = {{1}};

        for(int i=1; i<=n;i++){
            vector<int> re;
            re.push_back(1);

            for(int j=1; j<i;j++){
                re.push_back(result[i-1][j-1] + result[i-1][j]);
            }

            re.push_back(1);
            result.push_back(re);
        }

        return result;
    }

    vector<int> NthRowPascalTriangle1(int n) { 
        vector<int> result;

        int str = 1;
        result.push_back(str);

        for (int i = 1; i <= n; i++) {
            int curr = (str * (n - i + 1)) / i;

            result.push_back(curr);
            str = curr;
        }

        return result;
    }
    vector<int> NthRowPascalTriangle2(int n){
        vector<int> result;
        result.push_back(1);

        if(n==0) return result;

        vector<int> mid = NthRowPascalTriangle2(n-1);

        for(int i=1; i<mid.size(); i++){
            result.push_back(mid[i-1]+mid[i]);
        }

        result.push_back(1);

        return result;
    }

    bool isArmstrongNumber(int n){
        int digits=0,
            d = n,
            res = n;

        while(d){
            digits++; d/=10;
        }

        int arm=0;

        while(n){
            arm+= pow(n%10,digits);
            n/=10;
        }

        return arm == res;
        
    }

    double DeterminantOfMatrixRecursion(vector<vector<double>> matrix){
        int n =  matrix.size();

        if(n==1) return matrix[0][0];
        if(n==2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        
        double det = 0;

        for(int col = 0; col<n;col++){
            vector<vector<double>> subMatrix(n-1,vector<double>(n-1));
            
            for(int row = 1; row<n; row++){
                int count =0;

                for(int subCol =0; subCol<n; subCol++){

                    if(subCol== col) continue;
                    subMatrix[row-1][count] = matrix[row][subCol];
                }
            }

            det+=(col%2==0 ? 1:-1)*matrix[0][col]*DeterminantOfMatrixRecursion(subMatrix);
        }

        return det;
    }
    
    long long x_Pow_a_Mod_p1(int x, int a, int p){
        long long res = 1;
        x = x%p;

        while(a--){
            res *=x;

            if(res == p) return 0;
            if(res>p) res %=p;
        }

        return res;
    }
    long long x_Pow_a_Mod_p2(int x, int a, int p){
        if(a == 1) return x%p;
        return ((x%p)*(x_Pow_a_Mod_p2(x,a-1,p))%p)%p;
    }
    long long x_Pow_a_Mod_p3(int x, int a, int p){
        long long res= 1;
        x=x%p;

        while(a>0){
            if(a & 1) res = (res*x)%p;

            a = a>>1;
            x = (x*x)%p;
        }

        return res;
    }

    int LargeNumModNum(string num, int n){
        int res = 0;

        for(char digit:num){
            res = (10*res + digit - '0')%n;
        }

        return res;
    }

};

class Medium{
    public:

    bool IntegersLinearEquations(int a,int b, int c){
        while(max(a,b) % min(a,b) != 0){
            a = a%b;
            b= b%a;
        }

        if(c%min(a,b) == 0) return true;

        return false;
    }

    bool RelativelyPrime(int num1, int num2){
        while( min(num1, num2)  ){
            if(num1>num2) num1 %=num2;
            else num2 %=num1;
        }
       
        return !(max(num1,num2) - 1);
    }
    int NumRelativePrime(int n){
        int count=0;

        for(int i=1; i<n;i++)
            if(RelativelyPrime(i,n)) count++;

        return count;
    }
    int EulerTotientFunction(int n){
        float result = n;

        for(int i=2; i <= sqrt(n); i++){

            if(n%i == 0){

                while(n%i==0)
                    n/=i;
                result*=(1.0-(1.0/i));
            }
        }

        if(n>1) result -= result/n;

        return (int) result;
    }

    vector<int> SieveEratosthenes(int n){
        vector<int> sieve(n+1,1);
        sieve[0] = 0; sieve[1] = 0;

        for(int i=2; i<=sqrt(n); i++){

            if(sieve[i] == 1){
                int j= i*i;

                while(j <= n){
                    sieve[j]  = 0;j+=i;
                }
            }
        }

        vector<int> result;

        for(int i=2; i<=n ; i++)
            if(sieve[i]) result.push_back(i);

        return result;
    }

    vector<int> DivisorsNumber(int n){
        vector<int> result;

        for(int i=1; i<= n; i++)
            if(n%i == 0) result.push_back(i);

        return result;
    }

    vector<int> PrimeFactor(int n){
        vector<int> primes = SieveEratosthenes(n);
        vector<int> result;

        for(int prime:primes){

            while(n%prime == 0){
                result.push_back(prime);
                n/=prime;
            }

            if(n==1) break;
        }

        return result;
    }

    int LargestPrimeFactor(int n){
        vector<int> primes = SieveEratosthenes(n);

        for(int i=0; i<primes.size(); i++)
            if(n%primes[primes.size()-1-i] == 0) 
                return primes[primes.size() -1 -i];

        return 0;
    } 
    int MaxPrimeFactor(int n){
        int maxPrime = 0;
        
        while(n%2 == 0){
            maxPrime = 2;
            n/=2;
        }

        while (n%3 == 0){
            maxPrime = 3;
            n/=3;
        }
        
        for(int i=3; i*i <=n; i+=2){
            while(n%i== 0){
                maxPrime = i;
                n/=i;
            }
        }

        if(n>4) maxPrime = n;
        
        return maxPrime;
    }  

    string AddStrNum(vector<string> &nums){
        string result="";
        int n = nums.size();
        int len= nums[n-1].length();

        for(int i=0; i<n-1; i++){
            int num0 = len - nums[i].length();
            while(num0--){
                nums[i] = "0" + nums[i];
            }
        }

        int carry = 0;

        for(int i=len-1; i>=0; i--){
            int sum = 0;

            for(string num:nums){
                sum += num[i]- '0';
            }

            sum+=carry;
            carry=sum/10;
            int m = sum%10;
            result = string(1, (char)(m+'0')) + result;
        }

        if(carry>0){

            while(carry){
                int m=carry%10;
                result = string(1,(char)(m+'0')) + result;
                carry/=10;
            }
        }

        return result;
    } 
    string MulStrNum(string str, int num){
        vector<string> store;
        int count = 0;

        while(num){
            int n = num%10 ;
            string res="";
            int carry = 0;

            for(int i = str.size()-1;i>=0; i--){
                int z = (str[i]-'0')*n + carry;
                int m = z%10;

                res = string(1, (char)(m+'0') ) + res;    
                carry= z/10;
            }

            if(carry>0) res = string(1,(char)(carry+'0')) +res;

            for(int i =0; i<count; i++)
                res+="0";

            store.push_back(res);

            count++;
            num/=10;
        }
        return AddStrNum(store);
    }
    string FactorialLargeNum(int n){
        if(n==1) return "1";
        return MulStrNum(FactorialLargeNum(n-1), n);
    }
    
    void MulStrNumRemake(string &str, int num){
        int carry = 0;
        
        for(int i=0; i<str.length(); i++){
            int n = (str[i] - '0')*num +carry;
            
            str[i] = (char) ('0' + n%10);
            carry = n/10;
        }

        while(carry>0){
            str.push_back( (char) ('0' + carry%10));
            carry/=10;
        }

        return;
    }
    string FactorialLargeNumRemake(int n){
        string result = "1";
        
        for(int i=2; i<=n; i++)
            MulStrNumRemake(result, i);

        reverse(result.begin(), result.end());
        return result;
    }

    int LargestPowerkInN(int n, int k){
        int count=0;
        int fac = 1;

        for(int i=1; i<=n;  i++){
            fac*=i;

            if(fac%k == 0){
                count++;
                fac/=k;
            }
        }

        return count;
    }

    int LastNonZeroDigitFactorial(int n){
        int fac = 1;

        for(int i=1; i<=n; i++){
            fac*=i;

            while(fac%10==0)
                fac/=10;

            if(fac>1000) fac = fac % 10;
        }

        return fac%10;
    }

    vector<string> PowerSet(string s){
        int n= s.length();
        vector<string> result;

        for(int i = 0; i< pow(2,n); i++){
            string sub ="";

            for(int j= 0; i<n; j++){
                if( i & (1<<j)) sub+=s[j];
            }

            result.push_back(sub);
        }
        return result;
    }

    set<string> PermutationsString1(string s, int index){
        set<string> result;

        if(index == s.length())
            result.insert(s);

        else{

            for(int i=0; i<s.length(); i++){
                swap(s[index],s[i]); 
                set<string> recursion = PermutationsString1(s, index+1);

                for(string re:recursion){
                    result.insert(re);
                }

                swap(s[index],s[i]);
            }
        }
        return result;
    }
    set<string> PermutationsString(string s){
        return PermutationsString1(s, 0);
    }

    void SortMini(vector<int> &nums, int pv, int mv){
    
        for(int i=mv; i>pv;i--){

            if(nums[pv]<nums[i]){
                swap(nums[pv],nums[i]);
                break;
            }
        }

        for(int i=pv+1, j=mv; i<j; i++, j--){
            swap(nums[i],nums[j]);
        }
        return;
        
    }
    vector<int> NextPermutation(vector<int> &nums){
        int n = nums.size();
        int pv = n-2;
        
        while( pv>=0&& nums[pv] >= nums[pv+1]){
            pv--;
        }
        
        if(pv+1 == 0) {
            vector<int> result;

            for(int i=0; i<n;i++){
                result.push_back(nums[n-1-i]);
            }

            return result;
        }
        
        SortMini(nums,pv,n-1);
        return nums;

    }

    long long x_Pow_a_Mod_p(int x,int a, int p){
        long long res= 1;
        x=x%p;

        while(a>0){
            if(a & 1) res = (res*x)%p;

            a = a>>1;
            x = (x*x)%p;
        }

        return res;
    }
    bool isCarmichaelNum(int n){
        vector<int> relaPri;

        for(int i=1; i<n; i++)
            if(RelativelyPrime(i, n)) relaPri.push_back(i);

        for(int x:relaPri)
            if(x_Pow_a_Mod_p(x,n-1,n) != 1) return false;

        return true;
    }

    vector<int> CollatzSequence(int n){
        vector<int> result;
        result.push_back(n);

        while(n!=1) {
            if(n%2==0) n/=2;
            else n = 3*n+1;

            result.push_back(n);
        }

        return result;
    }

    int CountPathsRe(int m, int n){
        if(m == 1 || n == 1) return 1;
        return CountPathsRe(m-1,n) + CountPathsRe(m, n-1);
    }
    int CountPaths(int m, int n){
        int cp[m][n];
        cp[0][0] = 0;

        for(int i =1;i<m;i++) cp[i][0] = 1;
        for(int i =1;i<n;i++) cp[0][i] = 1;

        for(int im=1; im < m; im ++)
            for(int in=1; in< n; in++)
                cp[im][in] = cp[im-1][in] + cp[im][in-1];
        
        return cp[m-1][n-1];

    }
};

class Hard{
    public:

    int Josephus1(int n, int k){
        if( n == 1) return 0;
        return (Josephus1(n-1,k)+k) % n;
    }
    int Josephus2(int n, int k){
        vector<int> flag(n);

        for(int i=0; i<n ; i++){
            flag[i] = i+1;
        }

        int id = 0;

        while(flag.size() > 1){
            id = (id+k-1) % flag.size();
            flag.erase(flag.begin()+id);
            n--;
        }

        return flag[0];
    }
    int Josephus3(int n, int k){
        int cnt = 1,
            res = 0;

        while(cnt<=n){
            res = (res+k)%cnt;
            cnt++;
        }

        return res+1;
    } 

    vector<int> EratosthenesSieve(int n){
        vector<int> sieve(n+1,1);
        sieve[0] = 0; sieve[1] = 0;

        for(int i=2; i<=sqrt(n); i++){

            if(sieve[i] == 1){
                int j= i*i;

                while(j <= n){
                    sieve[j]  = 0;j+=i;
                }
            }
        }

        vector<int> result;

        for(int i=2; i<=n ; i++)
            if(sieve[i]) result.push_back(i);

        return result;
    }
    vector<int> SegmentedSieve(int n){
        vector<int> primes = EratosthenesSieve((int) sqrt(n));
        vector<int> result = primes;

        int lm = sqrt(n) + 1;
        int low = lm;
        int high=2*lm;

        while(low < n){
            if(high>n) high = n;

            vector<bool> mark(lm+1, 1);

            for(int i=0; i<primes.size(); i++){
                
                int minLm = (low- 1 + primes[i])/primes[i]*primes[i];

                for(int j=minLm; j<high; j+=primes[i])
                    mark[j-low] = 0;
            } 

            for( int i=low; i<high; i++)
                if(mark[i-low]) result.push_back(i);
            
            low +=lm;
            high+=lm;
        }

        return result;
    }

    int kThPrimeFactorNum(int n, int k){
        vector<int> primes = EratosthenesSieve(n);
        int id=0;

        while(id <primes.size()){

            if(n%primes[id] == 0){

                while( n%primes[id] == 0){
                    n/=primes[id];
                    k--;
                    if(k==0) return primes[id];
                }
            }

            else id++;
        }
        return -1;
    }

    void MultiplyLargeNum(vector<int> &nums, int x){
        int carry = 0;

        for(int i=0 ; i<nums.size(); i++){
            long long digit = (long long) nums[i]*x + carry;
            
            nums[i] = digit%10;
            carry = digit/10;
        }
        
        while(carry>0){
            nums.push_back(carry%10);
            carry/=10;
        }

        return;
    }
    int SumDigitsFactorial(int n){
        vector<int> fac{1};

        for(int i=2; i<=n; i++)
            MultiplyLargeNum(fac, i);
        
        int sum = 0;

        for(int digit:fac)
            sum+=digit;
        
        return sum;
    }

    int DropEgg(int n, int fl){
        if(fl <2 || n == 1) return fl;

        int minDrop = INT_MAX, res;  

        for(int flo = 1; flo <= fl; flo++){
            res = max(DropEgg(n-1, flo -1), DropEgg(n,fl - flo));
            if(res < minDrop) minDrop = res;
        }
        
        return minDrop + 1;
    }
    int MinTrials(int n, int fl){
        vector<int> dp(n+1,0);
        int test = 0;

        while(dp[n] < fl){
            test++;

            for(int x=n; x>0; x--)
                dp[x] = dp[x] + dp[x-1] + 1;
        }

        return test;
    }
    int MinTest(int n, int fl){
        int dp[fl+1][n+1] = {0};
        int test = 0;

        while(dp[test][n] < fl){
            test++;

            for(int x=1; x<=n; x++)
                dp[test][x] = dp[test-1][x-1] + dp[test][x-1] + 1;
            
        }

        return test;
    }

    void NextWord(string &s){
        if(s == ""){ 
            s = "a";
            return;
        }

        for(int i = s.length() - 1; i>=0; i--){
            if(s[i] != 'z') {
                s[i]++;
                return ;
            }
        }

        s[s.length()-1] = 'a';
        return;
    }

};

int main()
{

    Basic  bs;
    Easy   ez;
    Medium md;
    Hard   hd;

}   

