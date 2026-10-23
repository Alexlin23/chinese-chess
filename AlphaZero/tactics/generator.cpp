// Offline independent move generator and AND/OR mate prover. No neural labels.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
using Board = std::array<int8_t,90>;
struct Move { int f,t; };
struct State { Board b; int side=1, ply=0, nocap=0; std::vector<Board> hist; };
static std::mt19937_64 rng;
int ri(int n) { return int(rng()%n); }
bool palace(int r,int c,int s) { return c>=3&&c<=5&&(s==1?r>=7:r<=2); }
int king(const Board& b,int s) { for(int i=0;i<90;i++) if(b[i]==s) return i; return -1; }
int between(const Board& b,int f,int t) {
 int fr=f/9,fc=f%9,tr=t/9,tc=t%9;
 if(fr!=tr&&fc!=tc) return -1;
 int d=fr==tr?(tc>fc?1:-1):(tr>fr?9:-9), n=0;
 for(int i=f+d;i!=t;i+=d) n+=b[i]!=0;
 return n;
}
bool attacks(const Board& b,int f,int t,bool capture) {
 int p=std::abs(b[f]),s=b[f]>0?1:-1,fr=f/9,fc=f%9,tr=t/9,tc=t%9;
 int dr=tr-fr,dc=tc-fc,ar=std::abs(dr),ac=std::abs(dc);
 if(f==t) return false;
 switch(p) {
 case 1: return (ar+ac==1&&palace(tr,tc,s)) || (capture&&b[t]==-s&&dc==0&&between(b,f,t)==0);
 case 2: return ar==1&&ac==1&&palace(tr,tc,s);
 case 3: return ar==2&&ac==2&&(s==1?tr>=5:tr<=4)&&b[(fr+dr/2)*9+fc+dc/2]==0;
 case 4: return ((ar==2&&ac==1)||(ar==1&&ac==2))&&b[(fr+(ar==2?dr/2:0))*9+fc+(ac==2?dc/2:0)]==0;
 case 5: return between(b,f,t)==0;
 case 6: return between(b,f,t)==(capture?1:0);
 case 7: return (dr==-s&&dc==0) || (dr==0&&ac==1&&(s==1?fr<=4:fr>=5));
 }
 return false;
}
bool check(const Board& b,int s) {
 int k=king(b,s); if(k<0) return true;
 for(int i=0;i<90;i++) if(b[i]*s<0&&attacks(b,i,k,true)) return true;
 return false;
}
Board moved(Board b,Move m) { b[m.t]=b[m.f];b[m.f]=0;return b; }
std::vector<Move> legal(const Board& b,int s) {
 std::vector<Move> ms; if(king(b,s)<0||king(b,-s)<0) return ms;
 for(int f=0;f<90;f++) if(b[f]*s>0) for(int t=0;t<90;t++)
  if(b[t]*s<=0&&attacks(b,f,t,b[t]!=0)&&!check(moved(b,{f,t}),s)) ms.push_back({f,t});
 return ms;
}
bool any_legal(const Board& b,int s) {
 if(king(b,s)<0||king(b,-s)<0) return false;
 for(int f=0;f<90;f++) if(b[f]*s>0) for(int t=0;t<90;t++)
  if(b[t]*s<=0&&attacks(b,f,t,b[t]!=0)&&!check(moved(b,{f,t}),s)) return true;
 return false;
}
State apply(const State& x,Move m) {
 State y=x; y.b=moved(x.b,m);y.side=-x.side;y.ply++;y.nocap=x.b[m.t]?0:x.nocap+1;
 y.hist.insert(y.hist.begin(),y.b);if(y.hist.size()>8)y.hist.pop_back();return y;
}
bool mate_after(const Board& b,int s,Move m) {
 Board y=moved(b,m); return king(y,-s)<0 || !any_legal(y,-s);
}
std::vector<Move> mates1(const Board& b,int s,bool first=false) {
 std::vector<Move> answer;
 for(auto m:legal(b,s)) if(mate_after(b,s,m)) {answer.push_back(m);if(first)break;}
 return answer;
}
std::vector<Move> mates3(const Board& b,int s) {
 std::vector<Move> answer;
 for(auto m:legal(b,s)) {
  Board y=moved(b,m);
  auto replies=legal(y,-s); if(replies.empty())continue;
  bool proven=true;
  for(auto reply:replies) {
   Board z=moved(y,reply);
   if(king(z,s)<0 || mates1(z,s,true).empty()) {proven=false;break;}
  }
  if(proven)answer.push_back(m);
 }
 return answer;
}
int action(Move m,int side) {
 if(side<0) {m.f=89-m.f;m.t=89-m.t;}
 int dr=m.t/9-m.f/9,dc=m.t%9-m.f%9,p=-1;
 if(dc==0&&dr<0)p=-dr-1;
 else if(dc==0&&dr>0)p=9+dr-1;
 else if(dr==0&&dc<0)p=18-dc-1;
 else if(dr==0&&dc>0)p=26+dc-1;
 else {
  static const int ds[16][2]={{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1},{-2,-2},{-2,2},{2,-2},{2,2},{-1,-1},{-1,1},{1,-1},{1,1}};
  for(int i=0;i<16;i++)if(ds[i][0]==dr&&ds[i][1]==dc)p=34+i;
 }
 if(p<0)throw std::runtime_error("unencodable move");
 return p*90+m.f;
}
uint64_t board_key(const State& x) {
 Board a{},b{};
 for(int r=0;r<10;r++)for(int c=0;c<9;c++) {
  int v=x.side>0?x.b[r*9+c]:-x.b[(9-r)*9+8-c];a[r*9+c]=v;b[r*9+8-c]=v;
 }
 const auto& canonical=a<b?a:b; uint64_t h=1469598103934665603ULL;
 for(auto v:canonical){h^=uint8_t(v+8);h*=1099511628211ULL;}return h;
}
#pragma pack(push,1)
struct Record {
 int8_t history[8][90]{};uint8_t repetitions[8]{};
 int8_t turn=1;uint8_t history_length=1,category=0,family=0;
 uint16_t ply=0,no_capture=0,count=0,legal_count=0,actions[128]{},legal_actions[128]{};
 float probabilities[128]{},value=0,policy_weight=1,value_weight=0;
 uint64_t key=0;
};
#pragma pack(pop)
static_assert(sizeof(Record)==1784,"binary schema mismatch");
bool place(Board& b,int p,bool full_board=false) {
 for(int tries=0;tries<100;tries++) {
  int s=p>0?1:-1,t=std::abs(p),r=0,c=0;
  if(t==2) {static const int q[5][2]={{9,3},{9,5},{8,4},{7,3},{7,5}};int k=ri(5);r=q[k][0];c=q[k][1];if(s<0){r=9-r;c=8-c;}}
  else if(t==3) {static const int q[7][2]={{9,2},{9,6},{7,0},{7,4},{7,8},{5,2},{5,6}};int k=ri(7);r=q[k][0];c=q[k][1];if(s<0){r=9-r;c=8-c;}}
  else if(t==7) {r=ri(7);c=r>=5?2*ri(5):ri(9);if(s<0){r=9-r;c=8-c;}}
  else {r=ri(full_board?10:8);c=ri(9);if(!full_board&&s<0&&ri(4)==0)r=ri(10);}
  if(b[r*9+c]==0){b[r*9+c]=p;return true;}
 }
 return false;
}
State candidate(int family,bool dense) {
 State x; x.b.fill(0);x.side=1;x.ply=ri(130);
 x.b[(7+ri(3))*9+3+ri(3)]=1;x.b[ri(3)*9+3+ri(3)]=-1;
 static const std::vector<std::vector<int>> armies={{5,5},{5,4},{5,6},{6,6,7},{4,6,7},{5,7,7},{4,4,6},{5,6,7},{5,4,7},{5,6,4},{5,5,7},{6,6,4}};
 for(int p:armies[family])place(x.b,p);
 int count=1+ri(4);std::array<int,8> used{};
 for(int j=0;j<count;j++) {static const int ds[]={2,2,3,4,5,6,7,7};int p=ds[ri(8)];if(used[p]<(p==7?5:2)){place(x.b,-p);used[p]++;}}
 if(ri(2))place(x.b,ri(2)?2:3);
 if(dense) {
  // Bridge to actual games: legal piece inventories, 12--32 pieces in total.
  // Extra material may block, refute or create a mate; always re-prove afterward.
  for(int side: {1,-1}) {
   int desired=6+ri(11),have=0;std::array<int,8> inventory{};
   for(int p:x.b)if(p*side>0){inventory[std::abs(p)]++;have++;}
   for(int tries=0;have<desired&&tries<200;tries++) {
    int p=2+ri(6),limit=p==7?5:2;
    if(inventory[p]<limit&&place(x.b,side*p,true)){inventory[p]++;have++;}
   }
  }
 }
 x.hist={x.b};
 if(check(x.b,-x.side)||check(x.b,x.side)||legal(x.b,x.side).empty()){x.hist.clear();return x;}
 // Both players and every retained history length must occur in every class.
 // An even-length walk followed by a losing child would leak value via parity.
 int walk=8+ri(8);
 for(int i=0;i<walk;i++) {
  auto ms=legal(x.b,x.side);if(ms.empty()){x.hist.clear();return x;}
  State next=apply(x,ms[ri(ms.size())]);
  if(std::find(x.hist.begin(),x.hist.end(),next.b)!=x.hist.end()){x.hist.clear();return x;}
  x=next;
 }
 if(!any_legal(x.b,x.side)||check(x.b,-x.side))x.hist.clear();
 return x;
}
int main(int argc,char**argv) {
 if(argc!=5&&argc!=6){std::cerr<<"out count seed ood [dense]\n";return 2;}
 std::string path=argv[1];int target=std::stoi(argv[2]);rng.seed(std::stoull(argv[3]));bool ood=std::stoi(argv[4]);
 int density_mode=argc==6?std::stoi(argv[5]):0;
 std::ofstream out(path,std::ios::binary);if(!out)return 3;
 std::array<int,6> counts{},quotas{};double proportions[]={.30,.30,.15,.10,.10,.05};
 int total=0;for(int i=0;i<6;i++){quotas[i]=int(target*proportions[i]);total+=quotas[i];}quotas[0]+=target-total;
 total=0;uint64_t attempts=0;std::unordered_set<uint64_t> seen;auto started=std::chrono::steady_clock::now();
 bool dense_source=false;int density_counts[2][6]{};
 auto emit=[&](State x,int cat,int fam,const std::vector<Move>& moves,float v,float vw,float pw){
  if(counts[cat]>=quotas[cat]||moves.empty()||moves.size()>128)return;
  if(density_mode==2&&density_counts[dense_source][cat]>=quotas[cat]/2+(!dense_source?quotas[cat]%2:0))return;
  auto key=board_key(x);if(!seen.insert(key).second)return;
  // Colour-swapped rotation preserves the position; both absolute turn planes are learned.
  bool flipped=ri(2);
  if(flipped){for(auto& b:x.hist){Board y{};for(int i=0;i<90;i++)y[89-i]=-b[i];b=y;}x.b=x.hist[0];x.side=-x.side;}
  // Import a genuine short legal window, sampled independently of its label.
  // Rule clocks start at that imported window's oldest position.
  x.hist.resize(1+ri(x.hist.size()));x.ply=ri(130)+int(x.hist.size())-1;x.nocap=0;
  for(int i=int(x.hist.size())-2;i>=0;i--) {
   int before=0,after=0;for(int j=0;j<90;j++){before+=x.hist[i+1][j]!=0;after+=x.hist[i][j]!=0;}
   x.nocap=after<before?0:x.nocap+1;
  }
  Record r;r.turn=x.side;r.history_length=x.hist.size();r.category=cat;r.family=fam;r.ply=x.ply;r.no_capture=x.nocap;r.key=key;
  for(size_t i=0;i<x.hist.size();i++){
   std::copy(x.hist[i].begin(),x.hist[i].end(),r.history[i]);
   for(size_t j=i;j<x.hist.size();j+=2)r.repetitions[i]+=x.hist[j]==x.hist[i];
  }
  r.count=moves.size();r.value=v;r.value_weight=vw;r.policy_weight=pw;
  auto all=legal(x.b,x.side);if(all.size()>128)throw std::runtime_error("too many legal actions");
  r.legal_count=all.size();for(size_t i=0;i<all.size();i++)r.legal_actions[i]=action(all[i],x.side);
  for(size_t i=0;i<moves.size();i++) {
   Move m=moves[i];if(flipped){m.f=89-m.f;m.t=89-m.t;}
   // Moves were described before rotating the board; action() expects absolute coordinates.
   r.actions[i]=action(m,x.side);r.probabilities[i]=1.f/moves.size();
  }
  out.write(reinterpret_cast<const char*>(&r),sizeof(r));counts[cat]++;density_counts[dense_source][cat]++;total++;
 };
 while(total<target) {
  attempts++;int fam=ood?10+ri(2):ri(10);dense_source=density_mode==1||(density_mode==2&&ri(2));
  State x=candidate(fam,dense_source);if(x.hist.empty())continue;
  auto one=mates1(x.b,x.side);
  if(!one.empty()) {
   emit(x,0,fam,one,1,1,1);
  } else if(counts[1]<quotas[1]||counts[2]<quotas[2]) {
   // Enumerate every first move, including quiet moves, and every legal defensive reply.
   auto three=mates3(x.b,x.side);
   if(!three.empty()) {
    emit(x,1,fam,three,1,1,1);
    State d=apply(x,three[ri(three.size())]);
    emit(d,2,fam,legal(d.b,d.side),-1,1,0);
   }
  }
  // Find defence in the actual legally reached side-to-move, rather than
  // flipping a position's turn and exposing a one-frame category shortcut.
  if(counts[3]<quotas[3]&&one.empty()&&!mates1(x.b,-x.side,true).empty()) {
   std::vector<Move> safe,bad;
   for(auto m:legal(x.b,x.side)) {
    auto y=moved(x.b,m);(mates1(y,-x.side,true).empty()?safe:bad).push_back(m);
   }
   if(!safe.empty()&&!bad.empty())emit(x,3,fam,safe,0,0,1);
  }
  if(counts[4]<quotas[4]||counts[5]<quotas[5]) {
   std::vector<Move> captures,checks;
   for(auto m:legal(x.b,x.side)) {
    int p=std::abs(x.b[m.t]);if(p==4||p==5||p==6)captures.push_back(m);
    if(check(moved(x.b,m),-x.side))checks.push_back(m);
   }
   if(one.empty()) {emit(x,4,fam,captures,0,0,.25f);emit(x,5,fam,checks,0,0,.15f);}
  }
  if(attempts%1000==0 || total==target) {
   double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
   std::ofstream progress(path+".progress.json");progress<<"{\"accepted\":"<<total<<",\"attempts\":"<<attempts<<",\"seconds\":"<<sec<<",\"counts\":[";
   for(int i=0;i<6;i++)progress<<(i?",":"")<<counts[i];
   progress<<"]}";
  }
 }
 out.close();std::cout<<"generated="<<total<<" attempts="<<attempts<<" bytes="<<total*sizeof(Record)<<std::endl;
}
