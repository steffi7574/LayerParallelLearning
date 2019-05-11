
/*!
 * \class CFreeFormBlending
 * \brief Class that defines the particular kind of blending function for the free form deformation.
 * \author T. Albring
 */
class CFreeFormBlending {

protected:
  unsigned short Order, /*!< \brief Order of the polynomial basis. */
   Degree,              /*!< \brief Degree (Order - 1) of the polynomial basis. */
   nControl;            /*!< \brief Number of control points. */

public:

  /*!
   * \brief Constructor of the class.
   */
  CFreeFormBlending();

  /*!
   * \brief Destructor of the class.
   */
  virtual ~CFreeFormBlending();

  /*!
   * \brief A pure virtual member.
   * \param[in] val_i - index of the basis function.
   * \param[in] val_t - Point at which we want to evaluate the i-th basis.
   */
  virtual su2double GetBasis(short val_i, su2double val_t);

  /*!
   * \brief A pure virtual member.
   * \param[in] val_i - index of the basis function.
   * \param[in] val_t - Point at which we want to evaluate the derivative of the i-th basis.
   * \param[in] val_order - Order of the derivative.
   */
  virtual su2double GetDerivative(short val_i, su2double val_t, short val_order);

  /*!
   * \brief A pure virtual member.
   * \param[in] val_order - The new order of the function.
   * \param[in] n_controlpoints - the new number of control points.
   */
  virtual void SetOrder(short val_order, short n_controlpoints);

  /*!
   * \brief Returns the current order of the function.
   */
  su2double GetOrder();

  /*!
   * \brief Returns the current degree of the function.
   */
  su2double GetDegree();
};

/*!
 * \class CBSplineBlending
 * \brief Class that defines the blending using uniform BSplines.
 * \author T. Albring
 */
class CBSplineBlending : public CFreeFormBlending{

private:
  vector<su2double>          U;  /*!< \brief The knot vector for uniform BSplines on the interval [0,1]. */
  vector<vector<su2double> > N;  /*!< \brief The temporary matrix holding the j+p basis functions up to order p. */
  unsigned short KnotSize;       /*!< \brief The size of the knot vector. */

public:

  /*!
   * \brief Constructor of the class.
   */
  CBSplineBlending(short val_order, short n_controlpoints);

  /*!
   * \brief Destructor of the class.
   */
  ~CBSplineBlending();

  /*!
   * \brief Returns the value of the i-th basis function and stores the values of the i+p basis functions in the matrix N.
   * \param[in] val_i - index of the basis function.
   * \param[in] val_t - Point at which we want to evaluate the i-th basis.
   */
  su2double GetBasis(short val_i, su2double val_t);

  /*!
   * \brief Returns the value of the derivative of the i-th basis function.
   * \param[in] val_i - index of the basis function.
   * \param[in] val_t - Point at which we want to evaluate the derivative of the i-th basis.
   * \param[in] val_order - Order of the derivative.
   */
  su2double GetDerivative(short val_i, su2double val_t, short val_order_der);

  /*!
   * \brief Set the order and number of control points.
   * \param[in] val_order - The new order of the function.
   * \param[in] n_controlpoints - the new number of control points.
   */
  void SetOrder(short val_order, short n_controlpoints);

};

inline su2double CFreeFormBlending::GetBasis(short val_i, su2double val_t){return 0.0;}

inline su2double CFreeFormBlending::GetDerivative(short val_i, su2double val_t, short val_order){return 0.0;}

inline void CFreeFormBlending::SetOrder(short Order, short n_controlpoints){}

inline su2double CFreeFormBlending::GetOrder(){return Order;}

inline su2double CFreeFormBlending::GetDegree(){return Degree;}

